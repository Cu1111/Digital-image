import cv2
import numpy as np
import random

grey_img = cv2.imread("myGirl.jpg", 0)


def show_save_img(str, img):
    cv2.imshow(str, img)
    cv2.imwrite(str + ".png", img)


def adaptive_median_filter_step1(image, max_size=5):
    img = image.copy()
    border_width = int((max_size - 1) / 2)
    new_img = cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT,
                                 value=255)
    return new_img


def adaptive_median_filter(image, ori_img, row, col, kernel_size, max_size):
    padding = int((kernel_size - 1) / 2)
    margin = int((max_size - 1) / 2)
    new_row = row + margin
    new_col = col + margin
    new_img = image[new_row - padding:row + margin + padding, new_col - padding:new_col + padding]
    array = np.array(new_img)
    list = array.flatten().tolist()
    space = kernel_size ** 2
    list = list[:int((space - 1) / 2)] + list[int((space ** 2 - 1) / 2) + 1:space ** 2]
    if min(list) < np.median(list) < max(list):
        if min(list) < ori_img[row, col] < max(list):
            return ori_img[row, col]
        else:
            return np.median(list)
    else:
        kernel_size += 2
        if kernel_size <= max_size:
            return adaptive_median_filter(image, ori_img, row, col, kernel_size, max_size)
        else:
            return np.median(list)


def max_or_min_filter(image, mode):
    img = image.copy()
    rows, cols = image.shape
    new_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    for i in range(rows):
        for j in range(cols):
            roi_img = new_img[i:i + 3, j:j + 3].copy()
            array = np.array(roi_img)
            list = array.flatten().tolist()
            list = list[0:4] + list[5:9]
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi_img)
            # if img[i, j] < min_val or img[i, j] > max_val:
            #     if img[i, j] < min_val:
            #         img[i, j] = min_val
            #     else:
            #         img[i, j] = max_val
            if mode == 1:
                img[i, j] = max(list)
            else:
                img[i, j] = min(list)
    return img


def balance_noise(image, min=0, max=0.05):
    image = np.array(image / 255, dtype=float)
    noise = np.random.uniform(min, max, image.shape)
    out = image + noise
    print(out.min())
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    print(out.min())
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def main():
    cropped = grey_img[80:320, 600:920]
    show_save_img("cropped", cropped)
    noise1 = gasuss_noise(cropped)
    noise2 = sp_noise(noise1, 0.05)
    noise11 = sp_noise(cropped, 0.01)
    show_save_img("noise11", noise11)
    noise_img = balance_noise(noise2)
    show_save_img("noise_img", noise_img)
    blur_img = cv2.blur(noise_img, (3, 3))  # 均值滤波
    show_save_img("blur_img", blur_img)
    med_img = cv2.medianBlur(noise_img, 3)  # 中值滤波
    show_save_img("med_img", med_img)
    max_img = max_or_min_filter(noise11, 1)  # 最小值
    show_save_img("max_img", max_img)
    min_image = max_or_min_filter(noise11, 2)  # 最小值
    show_save_img("min_image", min_image)
    # 自适应中值滤波
    test_img = adaptive_median_filter_step1(noise_img)
    ada_img = noise_img.copy()
    rows, cols = ada_img.shape
    for i in range(rows):
        for j in range(cols):
            ada_img[i, j] = adaptive_median_filter(test_img, ada_img, i, j, 3, 5)
    show_save_img("ada_img", ada_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
