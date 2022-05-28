from utils import orientation
import cv2 as cv
import pandas as pd
import numpy as np
import time as t
import math
from PIL import Image
import utils.diagonal_crop


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv.warpAffine(image, M, (nW, nH))


def get_distance(img, p1, p2, img_name) -> int:
    import math
    # x = math.cos(1)
    # y = x * 180 / math.pi
    print("p1", p1)
    print("p2", p2)
    y = abs(p1[1] - p2[1])
    x = abs(p1[0] - p2[0])
    angle = y / x
    print("->", y, '/', x, '=', angle)
    # rad = angle * 180 / math.pi
    rad = angle
    angle = math.atan(rad)
    print("angle", math.degrees(angle))

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    im = Image.fromarray(img)
    angle = angle
    base = p1 if p1[0] < p2[0] else p2
    height = 20
    width = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
    cropped_im = utils.diagonal_crop.crop(im, base, angle, height, width)
    cont_lines = 0
    arr_gen = [1]
    # ant = 1  # 1: white, 0: black
    for x in range(cropped_im.width):
        for y in range(cropped_im.height):
            pixel = cropped_im.getpixel((x, y))
            if y == 0 and pixel == (0, 0, 0):
                cropped_im.putpixel((x, 1), (0, 0, 0))
            if y == 1:
                aux = None
                if pixel == (0, 0, 0):
                    aux = 0
                elif pixel == (255, 255, 255):
                    aux = 1
                if (aux is not None) and arr_gen[-1] != aux:
                    arr_gen.append(aux)

    cropped_im.save("cropped_"+img_name)
    cv.line(img, p1, p2, (255, 255, 0), 2)
    img = rotate_bound(img, math.degrees(angle))
    cv.imwrite(img_name, img)
    print("resultado vector")
    print(arr_gen)
    return arr_gen.count(0)


def process(x1, x2):
    def asignacion(df, centroids):
        colmap = {1: 'r', 2: 'g', 3: 'b'}
        for i in centroids.keys():
            # sqrt((x1 - c1)^2 - (x2 - c2)^2)
            df['distance_from_{}'.format(i)] = (
                np.sqrt(
                    (df['x1'] - centroids[i][0]) ** 2
                    + (df['x2'] - centroids[i][1]) ** 2
                )
            )
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
        df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
        df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        df['color'] = df['closest'].map(lambda x: colmap[x])
        return df

    def update(k):
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['closest'] == i]['x1'])
            centroids[i][1] = np.mean(df[df['closest'] == i]['x2'])
        return k

    t0 = t.time()
    colmap = {1: 'r', 2: 'g', 3: 'b'}

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2
    })

    np.random.seed(200)

    k = 3

    centroids = {
        i + 1: [np.random.randint(0, 296), np.random.randint(0, 560)]
        for i in range(k)
    }
    df = asignacion(df, centroids)

    centroids = update(centroids)
    df = asignacion(df, centroids)
    # print("time: ", t.time() - t0)
    # print(centroids)
    res = []
    for i in centroids.keys():
        if not math.isnan(centroids[i][0]):
            res.append((int(centroids[i][0]), int(centroids[i][1])))
    return res


def poincare_index_at(i, j, angles, tolerance):
    """
    compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
    https://books.google.pl/books?id=1Wpx25D8qOwC&lpg=PA120&ots=9wRY0Rosb7&dq=poincare%20index%20fingerprint&hl=pl&pg=PA120#v=onepage&q=poincare%20index%20fingerprint&f=false
    :param i:
    :param j:
    :param angles:
    :param tolerance:
    :return:
    """
    cells = [(-1, -1), (-1, 0), (-1, 1),  # p1 p2 p3
             (0, 1), (1, 1), (1, 0),  # p8    p4
             (1, -1), (0, -1), (-1, -1)]  # p7 p6 p5

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):

        # calculate the difference
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180

        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"


def calculate_singularities(im, angles, tolerance, W, mask):
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    colors = {"loop": (0, 0, 255), "delta": (0, 128, 255), "whorl": (255, 153, 255)}
    cont = 0
    loop_points_x1 = []
    loop_points_x2 = []
    delta_points_x1 = []
    delta_points_x2 = []
    for i in range(3, len(angles) - 2):  # Y
        for j in range(3, len(angles[i]) - 2):  # x
            # mask any singularity outside of the mask
            mask_slice = mask[(i - 2) * W:(i + 3) * W, (j - 2) * W:(j + 3) * W]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (W * 5) ** 2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none":
                    if singularity == "loop":
                        loop_points_x1.append(j * W)
                        loop_points_x2.append(i * W)
                    elif singularity == "delta":
                        delta_points_x1.append(j * W)
                        delta_points_x2.append(i * W)
                    cont += 1
                    font = cv.FONT_HERSHEY_SIMPLEX
                    org = (j * W, i * W)
                    fontScale = 0.5
                    thickness = 1
                    cv.putText(result, str(cont), org, font, fontScale, colors[singularity], thickness, cv.LINE_AA)
                    cv.circle(result, (j * W, i * W), 2, colors[singularity], -1)
    loop_res = process(loop_points_x1, loop_points_x2)
    print(loop_points_x1)
    print(loop_points_x2)
    for point in loop_res:
        cv.circle(result, point, 2, (0, 255, 0), -1)
    delta_res = process(delta_points_x1, delta_points_x2)
    cont = 0
    list_lines = []

    for l in loop_res:
        for d in delta_res:
            lines = get_distance(result, l, d, "{}.jpg".format(cont))
            list_lines.append(lines)
            cont += 1
    print("delta_res", delta_res)
    print("loop_res", loop_res)
    for point in delta_res:
        cv.circle(result, point, 2, (0, 0, 255), -1)
        # cv.circle(result, (point[1], point[0]), 2, colors["delta"], -1)
    # print((j*W), (i*W), singularity)
    # cont += 1
    # font = cv.FONT_HERSHEY_SIMPLEX
    # org = (j*W, i*W)
    # fontScale = 0.5
    # thickness = 1
    # cv.putText(result, str(cont), org, font, fontScale, colors[singularity], thickness, cv.LINE_AA)
    # cv.circle(result, (j*W, i*W), 2, colors[singularity], -1)
    ## cv.rectangle(result, ((j+0)*W, (i+0)*W), ((j+1)*W, (i+1)*W), colors[singularity], 3)

    return result, list_lines


if __name__ == '__main__':
    img = cv.imread('../test_img.png', 0)
    cv.imshow('original', img)
    angles = orientation.calculate_angles(img, 16, smoth=True)
    result, list_lines = calculate_singularities(img, angles, 1, 16)
