import cv2
import easyocr
import matplotlib.pyplot as plt


def pretty_print(text):
    for t in text:
        print(t)


def point_to_int(point):
    return (int(point[0]), int(point[1]))


def draw_square(img_, bbox):
    p0 = point_to_int(bbox[0])
    p1 = point_to_int(bbox[1])
    p2 = point_to_int(bbox[2])
    p3 = point_to_int(bbox[3])
    cv2.line(img_, p0, p1, (0, 255, 0), 1)
    cv2.line(img_, p1, p2, (0, 255, 0), 1)
    cv2.line(img_, p2, p3, (0, 255, 0), 1)
    cv2.line(img_, p3, p0, (0, 255, 0), 1)


def draw(text_, img_):
    threshold = 0.1

    for t in text_:
        bbox, reader_text, score = t

        if score > threshold:
            #cv2.rectangle(img_, bbox[0], bbox[2], (0, 255, 0), 5)
            draw_square(img_, bbox)
            cv2.putText(img_, reader_text, point_to_int(bbox[0]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1)

    plt.imshow(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
    plt.show()


# read image
image_path = "data/graph1.jpg"

img = cv2.imread(image_path)

# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text = reader.readtext(img)

pretty_print(text)

draw(text, img)
