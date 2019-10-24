import cv2
import numpy as np

grey_img = cv2.imread("myGirl.jpg", 0)
uequ_img = cv2.imread("uequimg.jpg", 0)
high_byte_image = np.copy(grey_img)
low_byte_image = np.copy(grey_img)
equ_image = np.copy(uequ_img)


def image_to_bit_plan_slicing(src):
    assert src.ndim == 2
    level = 8
    new_size = src.shape + (level,)
    bits_plan = np.unpackbits(src).reshape(new_size)
    for i in range(level):
        bits_plan[:, :, i] = np.bitwise_and(grey_img, 2 ** i)
    return bits_plan


def image_low_four(num):
    if len(bin(num)[2:]) <= 4:
        return num
    else:
        num_str = bin(num)[-4:]
        return eval("0b" + num_str)


def image_high_four(num):
    if len(bin(num)[2:]) <= 4:
        return 0
    else:
        num_str = bin(num)[:-4]
        return eval(num_str + "0000")


def equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]
    return img2


def main():
    # 转为灰色并显示
    global equ_image
    equ_image = equalization(equ_image)
    cv2.imwrite("equ_image.jpg", equ_image)
    cv2.imshow("origin", grey_img)
    grey_girl_bits = image_to_bit_plan_slicing(grey_img)
    # input_image_cp = np.where((input_image_cp >= 0) & (input_image_cp < 256), image_low_four(15), 0)
    for i in range(8):
        cv2.imshow('bit-plan-slicing-%d' % (i + 1), grey_girl_bits[:, :, i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 这里是高四位和第四位 已处理并保存
    # [rows, cols] = input_image_cp.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         high_byte_image[i, j] = image_high_four(high_byte_image[i, j])
    # cv2.imwrite("high_byte.jpg", high_byte_image)
    # for i in range(rows):
    #     for j in range(cols):
    #         low_byte_image[i, j] = image_low_four(low_byte_image[i, j])
    # cv2.imwrite("low_byte.jpg", low_byte_image)


if __name__ == '__main__':
    main()
