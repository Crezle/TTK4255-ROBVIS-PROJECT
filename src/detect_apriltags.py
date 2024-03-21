import cv2
print(cv2.__version__)

img = cv2.imread("data/apriltags/multiple_test/0007.jpg")

_, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters()
(corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

print(f"corners: {len(corners)}, rejected: {len(rejected)}")

color = (0, 255, 0)  # Green color

# Draw the IDs
if ids is not None:
    for i in range(len(ids)):
        c = corners[i][0]
        cv2.putText(img, "ID: " + str(ids[i][0]), (int(c[0][0]), int(c[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.aruco.drawDetectedMarkers(img, corners, ids)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

