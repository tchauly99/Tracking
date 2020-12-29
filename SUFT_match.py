import cv2
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def scale(matching_base, matching):
    sx = 0
    sy = 0
    count_x = 0
    count_y = 0
    # print(matching.shape[0])
    for i in range(matching.shape[0]):
        for j in range(i, matching.shape[0]):
            if i != j:
                sxi = abs((matching[i][0] - matching[j][0]) / (matching_base[i][0] - matching_base[j][0] + 0.9))
                syi = abs((matching[i][1] - matching[j][1]) / (matching_base[i][1] - matching_base[j][1] + 0.9))
                if 1.5 > sxi > 0.5:
                    count_x += 1
                    sx += sxi
                if 1.5 > syi > 0.5:
                    count_y += 1
                    sy += syi
    return (sx / count_x), (sy / count_y)


def inc(x1, x2, y1, y2):
    # z^2 + sum*z -0.1*mul = 0
    x = x2 - x1
    y = y2 - y1
    mul = x * y
    sum = x + y
    delta = sum * sum + 0.4 * mul
    if delta < 0:
        delta = 0
    z = (-sum + np.sqrt(delta)) / 2
    a = int(z / 2)
    x1 = x1 - a
    x2 = x2 + a
    y1 = y1 - a
    y2 = y2 + a
    return x1, x2, y1, y2, a


video = "walking.mp4"
cap = cv2.VideoCapture("walking.mp4")
first_frame = False
surf = cv2.xfeatures2d.SURF_create(100)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
first = True
second = False
third = False
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        first = False
        second = True
        third = False
        bb = cv2.selectROI("Frame", frame, fromCenter=False,
                           showCrosshair=True)
        x1 = bb[0]
        y1 = bb[1]
        x2 = x1 + bb[2]
        y2 = y1 + bb[3]
        img = frame[y1:y2, x1:x2]
        img2 = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # img2 = cv2.drawKeypoints(frame, kp, None, (255, 0, 0), 2)
    elif key == ord("q"):
        break
    if first:
        cv2.imshow("Frame", frame)
        continue
    if second:
        third = True
    else:
        x1, x2, y1, y2, a = inc(x1, x2, y1, y2)
        x = x2 - x1
        y = y2 - y1
    print("Target window")
    print(x1, x2, y1, y2)
    img = frame[y1:y2, x1:x2]
    kp, des = surf.detectAndCompute(img, None)
    img2 = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    kp_draw = []
    for i in range(len(kp)):
        kp_x = kp[i].pt[0] + x1
        kp_y = kp[i].pt[1] + y1
        kp_draw.append((int(kp_x), int(kp_y)))
        # cv2.circle(img2, (kp_draw[-1][0], kp_draw[-1][1]), 1, (255, 0, 255), 2)
        cv2.circle(img2, (int(kp_x), int(kp_y)), 1, (255, 0, 255), 2)
    cv2.imshow("Frame", img2)
    if second and third:
        des_base = des
        kp_base = kp
        x1_base = x1
        y1_base = y1
        weight_base = 15
        second = False
        continue
    matches = bf.match(des, des_base)
    # print("Number of keypoint base")
    # print(len(kp_base))
    # print("Number of keypoint")
    # print(len(kp))
    # print("Number of match")
    # print(len(matches))
    matching_base = []
    matching_des_base = []
    matching = []
    matching_des = []
    for match in matches:
        p_base = kp_base[match.trainIdx].pt
        p_base = (int(p_base[0] + x1_base), int(p_base[1] + y1_base))
        # p_base = (int(p_base[0]), int(p_base[1]))
        p = kp[match.queryIdx].pt
        p = (int(p[0] + x1), int(p[1] + y1))
        # p = (int(p[0]), int(p[1]))
        matching_base.append(p_base)
        matching_des_base.append(des_base[match.trainIdx])
        matching.append(p)
        matching_des.append(des[match.queryIdx])
    Mp = int((len(matches) / len(kp_base)) * 100)
    if Mp >= 20:
        # data_base = np.array(matching_base)
        # data_base = np.float32(data_base)
        # data = np.array(matching)
        # data = np.float32(data)

        data_base = np.array(matching_des_base)
        data_base = np.float32(data_base)
        pos_base = np.array(matching_base)
        data = np.array(matching_des)
        data = np.float32(data)
        pos = np.array(matching)

        _, label_base, _ = cv2.kmeans(data_base, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        _, label, _ = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        torso_base = []
        torso = []
        length_base = []
        length = []
        for i in range(3):
            torso_base.append(data_base[label_base.ravel() == i])
            torso.append(data[label.ravel() == i])
            # print(len(torso_base[-1]))
            length_base.append(len(torso_base[-1]))
            length.append(len(torso[-1]))
        torso_base_index = np.argmax(length_base)
        torso_index = np.argmax(length)
        # print("number of max base")
        # print(np.max(length_base))
        print("number of max keypoint of biggest region")
        print(np.max(length))
        # center_base = center_base[torso_base_index, :]
        # center = center[torso_index, :]
        # center = np.array(center)
        # data_biggest_cluster = data[label.ravel() == torso_index]
        # distance = data - center
        # distance = distance*distance
        # distance = np.sum(distance, axis=1)
        # distance = np.sqrt(distance)
        # center_index = np.argmin(distance, axis=0)
        # center = data[center_index]
        # center_base = data_base[center_index]
        torso_pos_base = pos_base[label_base.ravel() == torso_base_index]
        center_base = (np.sum(torso_pos_base, axis=0)[0] / torso_pos_base.shape[0]), \
                      (np.sum(torso_pos_base, axis=0)[1] / torso_pos_base.shape[0])

        torso_pos = pos[label.ravel() == torso_index]
        center = (np.sum(torso_pos, axis=0)[0] / torso_pos.shape[0]), \
                      (np.sum(torso_pos, axis=0)[1] / torso_pos.shape[0])

        # print(center[0], center[1])
        # for j, i in enumerate(data):
        #     if i[0] == center[0] and i[1] == [1]:
        #         matching_index = j

        # center_base = matching_base[matching_index]

        cv2.circle(img2, (int(center[0]), int(center[1])), 1, (0, 0, 255), 2)
        cv2.circle(img2, (int(center_base[0]), int(center_base[1])), 1, (0, 255, 255), 2)
        for i in range(len(matching)):
            cv2.circle(img2, (int(matching[i][0]), int(matching[i][1])), 1, (200, 0, 100), 2)
        cv2.imshow("Frame", img2)
        reposition_x = int(center[0] - center_base[0])
        reposition_y = int(center[1] - center_base[1])
        # reposition_x = reposition_x if abs(reposition_x) < 0.2 * x else int(reposition_x * 0.2 * x / abs(reposition_x))
        # reposition_y = reposition_y if abs(reposition_y) < 0.2 * y else int(reposition_y * 0.2 * y / abs(reposition_y))
        x1 = 0 if x1 + reposition_x < 0 else x1 + reposition_x
        x2 = 0 if x2 + reposition_x < 0 else x2 + reposition_x
        y1 = 0 if y1 + reposition_y < 0 else y1 + reposition_y
        y2 = 0 if y2 + reposition_y < 0 else y2 + reposition_y
        # x1 = x1 + reposition_x
        # x2 = x2 + reposition_x
        # y1 = y1 + reposition_y
        # y2 = y2 + reposition_y
        print("After Reposition")
        print(reposition_x, reposition_y)
        print(x1, x2, y1, y2)
        sx, sy = scale(pos_base[label.ravel() == torso_index],
                       pos[label.ravel() == torso_index])
        if sx * sy > 1.2 or sx * sy < 0.8:
            continue
        print("Scaling factor")
        print(sx, sy)
        kx = int((sx - 1) * (x2 - x1) / 2)
        ky = int((sy - 1) * (y2 - y1) / 2)
        x1 = x1 - kx
        x2 = x2 + kx
        y1 = y1 - ky
        y2 = y2 + ky
        print("After scale")
        print(x1, x2, y1, y2)
        x1 = x1 + a
        x2 = x2 - a
        y1 = y1 + a
        y2 = y2 - a
    else:
        print("TRACKING FAILED")
        break
    des_base = des
    kp_base = kp
    x1_base = x1
    y1_base = y1
    time.sleep(0.5)
