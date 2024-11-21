import numpy as np

# dt_boxes = np.array([[[6., 51.], [24., 57.], [19., 71.], [1., 65.]], [[4., 20.], [75., 23.], [74., 60.], [2., 57.]]])
dt_boxes = np.array([[[76., 86.], [88., 86.], [88., 94.], [76., 94.]],
                     [[12., 36.], [98., 40.], [97., 75.], [11., 71.]],
                     [[35., 27.], [101., 30.], [100., 40.], [35., 37.]],
                     [[36., 16.], [58., 16.], [58., 25.], [36., 25.]],
                     [[12., 15.], [27., 15.], [27., 32.], [12., 32.]],
                     [[54., 12.], [103., 14.], [103., 27.], [53., 25.]]])
position = 0


def getPolygonArea(dt_boxes):
    list1 = []
    for points in dt_boxes:
        area = points[-1][0] * points[0][1] - points[0][0] * points[-1][1]
        for i in range(1, len(points)):
            v = i - 1
            area += (points[v][0] * points[i][1])
            area -= (points[i][0] * points[v][1])
        list1.append(abs(0.5 * area))
    i = list1.index(max(list1))
    return list1, i


def getPolygonCertroid(points):
    x_sum, y_sum = 0., 0.
    for point in points:
        x_sum += point[0]
        y_sum += point[1]
    centroid = [x_sum / len(points), y_sum / len(points)]

    return centroid


def defineOtherPosition(dt_boxes, index, position):
    number_box = dt_boxes[index]
    number_centroid = getPolygonCertroid(number_box)

    other_boxes = np.delete(dt_boxes, index, axis=0)
    # print(other_boxes)
    top_left, top_right, bot_left, bot_right = np.array([number_box]), np.array([number_box]), np.array([number_box]), np.array([number_box])
    for point in other_boxes:
        point_centroid = getPolygonCertroid(point)
        if point_centroid[0] < number_centroid[0] and point_centroid[1] < number_centroid[1]:
            top_left = np.append(top_left, [point], axis=0)
        if point_centroid[0] > number_centroid[0] and point_centroid[1] < number_centroid[1]:
            top_right = np.append(top_right, [point], axis=0)
        if point_centroid[0] < number_centroid[0] and point_centroid[1] > number_centroid[1]:
            bot_left = np.append(bot_left, [point], axis=0)
        if point_centroid[0] > number_centroid[0] and point_centroid[1] > number_centroid[1]:
            bot_right = np.append(bot_right, [point], axis=0)
    if position == 0:
        return top_left
    if position == 1:
        return top_right
    if position == 2:
        return bot_left
    if position == 3:
        return bot_right


dict, index = getPolygonArea(dt_boxes)
# print(dict, index, dt_boxes[index])
# print(getPolygonCertroid(dt_boxes[index]))
print(defineOtherPosition(dt_boxes, index, position))
