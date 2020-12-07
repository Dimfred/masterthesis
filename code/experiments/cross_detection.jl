using Images, Colors, TestImages, ImageTransformations
using ImageView
function run()

    function nonmax_suppression(x::Matrix{T}) where {T}
        I, J = size(x)
        y = copy(x)
        for j = 1:J
            for i = 1:I
                c = x[i,j]
                for j2 = max(j - 1, 1):min(j + 1, J),
                    i2 = max(i - 1, 1):min(i + 1, I)
                    if x[i2, j2] > c
                        y[i, j] = 0
                        break
                    end
                end
            end
        end
        return y
    end

    #x = green(load("data/valid/00_20.jpg"))
    img = load("data/valid/00_20.jpg")
    #img = load("data/valid/07_05.png")
    img = Gray.(img)
    img = imresize(img, ratio=1/2)
    #using ImageView
    #imshow(img)
    img = imfilter(img, KernelFactors.gaussian((1.2, 1.2)))
    gx, gy = imgradients(img, KernelFactors.bickley)
    α1 = gx .* gx - gy .* gy
    α2 = 2 * gx .* gy
    len = 40
    α1 = imfilter(α1, KernelFactors.gaussian((1.2, 1.2)))
    α2 = imfilter(α2, KernelFactors.gaussian((1.2, 1.2)))
    h1 = [(i^2 - j^2) / max(1, i^2 + j^2) for i = -len:len , j = -len:len]
    h2 = [2 * i * j / max(1, i^2 + j^2) for i = -len:len , j = -len:len]
    r = imfilter(α2, centered(h2)) - imfilter(α1, centered(h1))
    r = nonmax_suppression(r)

    imshow((r .> 0) + img)
end
run()
