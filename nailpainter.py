import sys
import numpy as np
import tensorflow as tf
import cv2
from findfinger import find_nails, rgb

# Нейроная сеть
model = tf.Graph()
model_file = 'naildetector_model/frozen_inference_graph.pb'
min_confidence = 0.6

# Загрузка нейроной сети
with model.as_default():
    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.gfile.GFile(model_file, 'rb') as f:
        serialized_graph = f.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

# Сессия TF1 (с графом по умолчанию)
with model.as_default():
    with tf.compat.v1.Session(graph=model) as sess:
        image_tensor = model.get_tensor_by_name('image_tensor:0')
        boxes_tensor = model.get_tensor_by_name('detection_boxes:0')
        scores_tensor = model.get_tensor_by_name('detection_scores:0')

        # Веб-камера
        s = 0
        if len(sys.argv) > 1:
            s = sys.argv[1]

        source = cv2.VideoCapture(s)

        # Создание окна
        win_name = 'Nail Painter'
        cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
        cv2.createTrackbar('Red', win_name, 200, 255, rgb)
        cv2.createTrackbar('Green', win_name, 50, 255, rgb)
        cv2.createTrackbar('Blue', win_name, 100, 255, rgb)

        # Основной цикл программы
        while cv2.waitKey(1) != 27:  # Escape
            has_frame, frame = source.read()
            if not has_frame:
                break

            frame = cv2.flip(frame, 1)
            image = frame
            h, w = image.shape[:2]
            output = image.copy()

            # Предобработка кадра для нейросети
            result_image = find_nails(image.copy())
            image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            # Обработка кадра нейросетью
            boxes, scores = sess.run([boxes_tensor, scores_tensor], feed_dict={image_tensor: image})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            box_mid = (0, 0)

            # Покраска ногтей
            for box, score in zip(boxes, scores):
                if score > min_confidence:
                    (start_Y, start_X, end_Y, end_X) = box
                    start_X = int(start_X * w)
                    start_Y = int(start_Y * h)
                    end_X = int(end_X * w)
                    end_Y = int(end_Y * h)
                    X_mid = start_X + int(abs(end_X - start_X) / 2)
                    Y_mid = start_Y + int(abs(end_Y - start_Y) / 2)
                    box_mid = (X_mid, Y_mid)

                    sub_image = output.copy()
                    color = rgb(win_name)

                    cv2.ellipse(sub_image, box_mid, (int((end_Y - start_Y) * 0.6), int((end_X - start_X) * 0.5)),
                                0, 0, 360, color, -1)
                    cv2.addWeighted(sub_image, 0.75, output, 0.25, 0.0, output)

            cv2.imshow(win_name, output)

        source.release()
        cv2.destroyWindow(win_name)
