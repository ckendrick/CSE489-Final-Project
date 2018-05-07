import os
import os.path as path

from scipy import ndimage
import cv2

for root, dirs, files in os.walk("import", topdown=True):

    if dirs is not None:
        for dir in dirs:
            folder = 'export/{}'.format(dir)

            if not path.exists(folder):
                os.mkdir(folder)

            for root, dirs, files in os.walk('import/{}'.format(dir)):
                for i, f in enumerate(files):

                    img = ndimage.imread('import/{}/{}'.format(dir, f))
                    resized_image = cv2.resize(img, (200, 200))
                    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    # cv2.imshow('image', resized_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    cv2.imwrite('export/{}/{}'.format(dir, f), resized_image)
                    print('{}/{}'.format(i, files.__len__()))