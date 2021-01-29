import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

import time
from tabulate import tabulate

import numpy as np
import numba as nb


class GradientAccumulator:
    def __init__(self, accumulation_steps, trainable_vars):
        self.trainable_vars = trainable_vars
        self.accumulator = self.new_accumulator
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0

    @property
    def new_accumulator(self):
        return [tf.zeros_like(tv) for tv in self.trainable_vars]

    def accumulate(self, step_grads):
        self.accumulation_counter += 1

        self.accumulator = [
            (accu + grad) for accu, grad in zip(self.accumulator, step_grads)
        ]

        if (self.accumulation_counter % self.accumulation_steps) != 0:
            return None

        self.accumulation_counter = 0
        _accumulator = [accu / self.accumulation_steps for accu in self.accumulator]
        self.accumulator = self.new_accumulator

        return zip(_accumulator, self.trainable_vars)


class LossAccumulator:
    def __init__(self, accumulation_steps):
        self.accumulator = self.new_accumulator
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0

    def accumulate(self, losses):
        self.accumulation_counter += 1
        self.accumulator += losses

        if (self.accumulation_counter % self.accumulation_steps) != 0:
            return None

        self.accumulation_counter = 0
        _accumulator = np.array(
            [var / self.accumulation_steps for var in self.accumulator]
        )
        self.accumulator = self.new_accumulator
        return _accumulator

    @property
    def new_accumulator(self):
        return np.zeros((3,))


class LearningRateScheduler:
    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    def __call__(self, step, base_lr, burn_in):
        lr = self.schedule(step, base_lr, burn_in)
        K.set_value(self.model.optimizer.lr, lr)

    def __bool__(self):
        return self.schedule is not None


class Accumulative(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        optimizer,
        accumulation_steps=1,
        batch_size=1,
        name="Accumulative",
        **kwargs,
    ):
        super(Accumulative, self).__init__(name, **kwargs)

        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.batch_size = batch_size

        self.counter = 0
        self.gradient_accumulator = None

    def is_update_time(self):
        return (self.counter % (self.accumulation_steps * self.batch_size)) == 0

    def accumulate_grads(self, grads_and_vars):
        if self.gradient_accumulator is None:
            self.gradient_accumulator = [
                self._flat_gradients(g) / (self.accumulation_steps * self.batch_size)
                for g, _ in grads_and_vars
            ]
            return

        for i, (g, _) in enumerate(grads_and_vars):
            self.gradient_accumulator[i] += self._flat_gradients(g) / (
                self.accumulation_steps * self.batch_size
            )

    def _flat_gradients(self, grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
        if type(grads_or_idx_slices) == tf.IndexedSlices:
            return tf.scatter_nd(
                tf.expand_dims(grads_or_idx_slices.indices, 1),
                grads_or_idx_slices.values,
                grads_or_idx_slices.dense_shape,
            )

        return grads_or_idx_slices

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        self.counter += 1

        self.accumulate_grads(grads_and_vars)

        if self.is_update_time():
            print("APPLYING GRADIENTS")
            vars_ = (v for _, v in grads_and_vars)
            grad_zip = zip(self.gradient_accumulator, vars_)
            self.optimizer.apply_gradients(grad_zip)

            # reset accumulator
            self.gradient_accumulator = None

        print("APPLY CALLED")

    def get_config(self):
        iterations = self.iterations.numpy()
        self.iterations = 0
        config = self.optimizer.get_config()
        self.iterations = iterations
        return config


# class Accumulative(tf.keras.optimizers.Optimizer):
#     ################################################################
#     ### Code from https://github.com/LJNL/accum_optimizer_for_keras
#     ################################################################
#     def __init__(self, optimizer, accum_steps=1, name="accum", **kwargs):
#         self.name = name
#         super(Accumulative, self).__init__(name, **kwargs)
#         self.optimizer = optimizer
#         with tf.name_scope(self.__class__.__name__):
#             self.accum_steps = accum_steps
#             self.iterations = tf.Variable(0, dtype="int64", name="iterations")
#             self.cond = tf.equal(self.iterations % self.accum_steps, 0)
#             self.lr = self.optimizer.lr
#             self.optimizer.lr = tf.cond(
#                 self.cond, lambda: self.optimizer.lr.value(), lambda: 0.0
#             )
#             for attr in ["momentum", "rho", "beta_1", "beta_2"]:
#                 if hasattr(self.optimizer, attr):
#                     value = getattr(self.optimizer, attr)
#                     setattr(self, attr, value)
#                     setattr(
#                         self.optimizer,
#                         attr,
#                         tf.cond(self.cond, lambda: value.value(), lambda: 1 - 1e-7),
#                     )
#             for attr in self.optimizer.get_config():
#                 if not hasattr(self, attr):
#                     value = getattr(self.optimizer, attr)
#                     setattr(self, attr, value)

#             self._create_slots = self.optimizer._create_slots
#             self._resource_apply_dense = self.optimizer._resource_apply_dense

#             def get_gradients(loss, params):
#                 tf.print("accu")
#                 return [ag / self.accum_steps for ag in self.accum_grads]

#             self.optimizer.get_gradients = get_gradients

#     def get_updates(self, loss, params):
#         tf.print("optim", self.iterations)
#         self.iterations = tf.add(self.iterations, 1)
#         self.optimizer.iterations = tf.add(
#             self.optimizer.iterations, tf.cast(self.cond, "int64")
#         )
#         self.updates = [self.iterations, self.optimizer.iterations]
#         # gradient accumulation
#         self.accum_grads = [tf.zeros(p.shape, dtype=p.dtype) for p in params]
#         grads = self.get_gradients(loss, params)

#         for g, ag in zip(grads, self.accum_grads):
#             self.updates.append(ag=tf.cond(self.cond, lambda: g, lambda: ag + g))

#         # inheriting updates of original optimizer
#         self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
#         self.weights.extend(self.optimizer.weights)
#         return self.updates

#     def get_config(self):
#         iterations = self.iterations.numpy()
#         self.iterations = 0
#         config = self.optimizer.get_config()
#         self.iterations = iterations
#         return config

# def apply_gradients(self, grads_and_vars *args, name, experimental_aggregate_gradients:
#     print("DONOTHING")


class BatchProgbarLogger(callbacks.ProgbarLogger):
    def __init__(self, accumulation_steps=1, count_mode="steps", stateful_metrics=None):
        super(BatchProgbarLogger, self).__init__(count_mode, stateful_metrics)

        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        self.batch_counter = 0

        self.tloss_keys = ["loss", "output_1_loss", "output_2_loss", "output_3_loss"]
        self.tloss_accu = self.new_loss_accu()

        self.batch_time = time.perf_counter()

    def on_train_batch_end(self, batch, logs={}):
        print(logs)
        self.step_counter += 1

        self.tloss_accu = self.accumulate(self.tloss_accu, self.tloss_keys, logs)
        if (self.step_counter % self.accumulation_steps) == 0:
            self.batch_counter += 1

            batch_took = time.perf_counter() - self.batch_time
            self.batch_time = time.perf_counter()

            p = [["Batch", "Took", "LossSum", "LossLarge", "LossMedium", "LossSmall"]]
            p += [
                [
                    self.batch_counter,
                    f"{self.ffloat(batch_took)}s",
                    *(self.ffloat(l) for l in self.tloss_accu),
                ]
            ]
            tf.print(tabulate(p))

            # reset loss accu
            self.tloss_accu = self.new_loss_accu()

    def on_test_batch_end(self, batch, logs={}):
        tf.print(logs)

    def on_epoch_end(self, step, logs={}):
        # print(logs)
        pass

    def on_predict_end(self, step, logs):
        pass

    def on_epoch_begin(self, step, logs):
        pass

    def new_loss_accu(self):
        return [0 for _ in range(4)]

    def accumulate(self, accu, loss_keys, losses):
        for i, key in enumerate(loss_keys):
            accu[i] += losses[key] / self.accumulation_steps

        return accu

    def ffloat(self, f):
        return "{:.5f}".format(f)
