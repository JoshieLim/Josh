import cv2


def pad_image(image, size=640):
    """
    Apply resizing function and padding while keeping the aspect ratio

    Args:
        image: The image to be processed
        size: output image size. Default = (640x640)

    Returns:
        Processed image array
    """

    desired_size = size
    old_size = image.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT,
                                value=color)
    return new_im
