# Initial Structure

**\<SOMETHING\>** := optional / maybe

1. Introduction
    1. Motivation


    1. Related Work
        1. component
        1. component + location
        1. component + location + conversion

    1. Goals of the Thesis
        1. Task description (complete pipeline without solution)
            1. General (Img => ObjDetection / Seg => Build Topology => LTSpice output)
                - Different paper backgrounds
                - Diagram to show general pipeline

        1. Contribution
            - Grid paper
            - **Evaluation Algorithm**

1. Theory
    1. Electrical Circuit Diagrams

    1. LTSpice file
        - Structure
            - Grid propertys 
            - Wires
            - Components


    1. Neural Networks
        - MLP
        - MLP + hidden
        - Structure
        - Forward 
        - Activation Function
        - Backpropagation 
        - Optimization
        - Loss

    1. Convolutional Neural Networks
        - Convolutions

    1. Object detection
        1. General
            - Bounding Boxes

        1. **History**
            - Sliding Window
            - Region Proposal
            - Fast RCNN
            - Faster RCNN
            - Singleshot

        1. Yolov4
            - Objective
            - Backbone (Darknet53)
            - Neck (PAN)
            - Head (YoloHeadV3)
            - Loss functions: IoU, GIoU, DIoU, CIoU **experiments**

    1. Semantic Segmentation
        - Objective
        - UNet / MobileNetV2-UNet
        - TODO Indepth Architecture
        - Loss functions: Focal bla **experiments**

    1. Connected components

    1. OCR
        - Tesseract

    1. Graph matching theory (evaluation)
        - adjacency
        - hypergraph

    1. Proposed metrics // explain here or in the evaluation?

    1. Data Augmentation

    1. **BFS**

1. Materials and Methods
    1. Data
        1. Dataset labels bla / statistics

        1. Object Detection
            - Annotation Format
            - Used Software

        1. Segmentation
            - Annotation Format
            - Auto labeling procedure
                - dilated canny mask

        1. General 
            - Train / Val / Test Split 

    1. Pipeline
        - General

        1. Preprocessing
            1. Yolo detect
            1. MobileNetV2-UNet segment
            1. Multiply Segmask with bin-threshold of original img
        
        1. Postprocessing
            1. Topology building
                - Connected components (CCA)
                - Intersect CCA-output with BBox
                - Hypergraph Adjacency

            1. **Cosmetics**

    1. Hypergraph Conversion to LTSpice
        - projection from img space into LTSpice-space
        - mapping of components
        - converting 


    1. Training
        1. **Projection of annotations**
            - mnist
            - emnist? (with letters)

        1. Projection of gridded paper 
            - grabcut to extract masks
            - project masked data on grid paper 

        1. Yolo
            1. Augmentation
                - Rotation
                - RandomScale
                - Crop (SafeBBox Crop)
                - CLAHE
                - ColorJitter
                - Blur
                - GaussianNoise

            1. Hyper parameters
            1. Loss function experiments
            1. Metrics, Loss during training

        1. Mobile-UNet
            1. Augmentations
                - TODO augmentations

            1. Hyper parameters
            1. Loss function experiements
            1. Metrics, Loss during training


1. Results
    1. Yolo
        - mAP

    1. MobileNetV2-UNet
        - mIoU

    1. System
        - topology
            - macro 
            - micro
        
        - **annotation matching**
        

1. **Discussion**
    1. Yolo

    1. MobileNetV2-UNet
        - weak labels => bad segmentation results 

    1. Problems with holes in wire
        - train yolo to detect hoes

    1. Cosmetics 
        - BFS to find a path from node to node
        - Create fake nodes 
        - **Train yolo to detect fake nodes**

1. Conclusion

