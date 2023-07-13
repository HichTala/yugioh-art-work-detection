import math
import os

import cv2
import numpy as np


def get_lower_edges(coordinates):
    # Sort the coordinates in ascending order based on the y-coordinate
    sorted_coordinates = sorted(coordinates, key=lambda coord: coord[1])

    # Get the two lower vertices of the rectangle
    lower_vertices = sorted_coordinates[:2]

    # Calculate the center point of the rectangle
    center_x = sum(coord[0] for coord in coordinates) / 4
    center_y = sum(coord[1] for coord in coordinates) / 4

    # Sort the two lower vertices based on their angle with respect to the center
    sorted_lower_vertices = sorted(lower_vertices, key=lambda coord: math.atan2(coord[1] - center_y, coord[0] - center_x))

    # Get the indices of the sorted lower vertices
    index1 = coordinates.index(sorted_lower_vertices[0])
    index2 = coordinates.index(sorted_lower_vertices[1])

    # Determine the indices of the remaining vertices
    remaining_indices = [0, 1, 2, 3]
    remaining_indices.remove(index1)
    remaining_indices.remove(index2)

    # Determine the other two vertices of the rectangle
    other_vertices = [coordinates[remaining_indices[0]], coordinates[remaining_indices[1]]]

    # Determine the two lower edges of the rectangle
    lower_edge1 = sorted_lower_vertices[0], other_vertices[0]
    lower_edge2 = sorted_lower_vertices[1], other_vertices[1]

    return lower_edge1, lower_edge2


folder_dir = "ROI"
for file_name in os.listdir(folder_dir):
    if file_name.endswith(".png"):

        # Load image, grayscale, median blur, sharpen image
        image = cv2.imread(os.path.join(folder_dir, file_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        equalized = cv2.equalizeHist(gray)
        # blur = cv2.GaussianBlur(equalized, (7, 7), 0)
        _, thresh = cv2.threshold(equalized, 140, 255, cv2.THRESH_BINARY)
        # # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # # sharpen = cv2.filter2D(equalized, -1, sharpen_kernel)
        # gauss = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # canny = cv2.Canny(equalized, 100, 150)
        # # # Threshold and morph close
        # # _, thresh = cv2.threshold(canny, 50, 255, cv2.THRESH_BINARY_INV)
        # #
        kernel = np.ones((7, 7), np.uint8)
        edged = cv2.erode(thresh, kernel, iterations=3)
        edged = cv2.dilate(edged, kernel, iterations=3)
        # # # gray = np.float32(gray)
        # # # dst = cv2.cornerHarris(blur, 2, 3, 0.04)
        # # # result is dilated for marking the corners, not important
        # # # dst = cv2.dilate(dst, None)
        # # # Threshold for an optimal value, it may vary depending on the image.
        # # # image[dst > 0.04 * dst.max()] = [0, 0, 255]
        # #
        # # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # # # close = cv2.morphologyEx(gauss, cv2.MORPH_CLOSE, kernel, iterations=2)
        # # #
        # # # lines = cv2.HoughLinesP(gauss, 1, np.pi / 180, 25, minLineLength=0, maxLineGap=250)
        # # #
        # # # # Draw lines on the image
        # # # for line in lines:
        # # #     x1, y1, x2, y2 = line[0]
        # # #     cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # #
        # Find contours and filter using threshold area
        cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        cv2.drawContours(image, cnts, -1, (255, 229, 204), 2)
        min_area = 2000
        max_area = 8000
        image_number = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                print(cv2.contourArea(box))
                print(box)
                breakpoint()
                cv2.drawContours(image, [box], 0, (204, 229, 255), 2)

                # x, y, w, h = cv2.boundingRect(c)
                # ROI = image[y:y + h, x:x + w]
                # # cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
                # # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                # cv2.drawContours(image, c, -1, (204, 229, 255), 2)
                # image_number += 1

        # for cnt in cnts:
        #     approx = cv2.approxPolyDP(cnt)
        #         #     if len(approx) == 4:
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         ratio = float(w) / h
        #         if ratio >= 0.9 and ratio <= 1.1:
        #             cv2.putText('Square')
        #         else:
        #             cv2.putText('Rectangle')

        # cv2.imshow('gray', gray)
        # cv2.imshow('blur', blur)
        # cv2.imshow('equalized', equalized)
        # cv2.imshow('sharpen', sharpen)
        # cv2.imshow('close', close)
        # cv2.imshow('gauss', gauss)
        # cv2.imshow('canny', canny)
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('edged', edged)
        cv2.imshow('image', image)
        cv2.waitKey()
