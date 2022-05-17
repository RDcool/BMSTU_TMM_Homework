import numpy as np
import cv2


def find_nails(frame):
    """
    Функция для предобработки изображения для нейронной сети

    :param frame: Кадр для обработки
    :return: Обработанное изображение
    """
    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (3, 3), 0)

    mask = cv2.inRange(YCrCb_frame, np.array([0, 127, 75]), np.array([255, 177, 130]))
    bin_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=5)
    result = cv2.bitwise_and(frame, frame, mask=bin_mask)

    return result


def rgb(window):
    """
    Фунция для задания цвета ногтей
    :param window: Имя окна с ползунками
    :return: Цвет в формате BGR
    """
    r = cv2.getTrackbarPos('Red', window)
    g = cv2.getTrackbarPos('Green', window)
    b = cv2.getTrackbarPos('Blue', window)
    color = (b, g, r)
    return color
