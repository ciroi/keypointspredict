import numpy as np
import random
import cv2 as cv

IMAGE_PATH = "../image/"
LABEL_PATH = "../label/"
LABEL_SPLIT_PATH = "../label/"

negsample_ratio = 0.3
neg_gen_thre = 100
random_times = 3
random_border = 10

def GetItemData(allData, idex=None):
	if idex is None:
		idex = random.randint(0, len(allData))
	line = allData[idex]
	line = line.strip()
	return line.split()

def showImage(image, bound=None, keyPoints=None, expand=True):
	src = cv.imread(IMAGE_PATH+image)
	if bound:
		bound = [int(float(x)) for x in bound]
		cv.rectangle(src, (bound[0], bound[1]), (bound[2], bound[3]), (255, 0, 0), 1)

	for pt in [keyPoints[i:i+2]for i in range(0, len(keyPoints), 2)]:
		cv.circle(src, (int(float(pt[0])), int(float(pt[1]))), radius=2, color=(255, 255, 0), thickness=-2)
	
	cv.imshow("src",src)
	cv.waitKey(0)

def preShow():
	path = LABEL_SPLIT_PATH+"train_11.txt"
	fo = open(path, "r")
	allData = fo.readlines()
	allLen = len(allData)
	for idx in range(allLen):
		strData = GetItemData(allData, idx)
		showImage(strData[0], strData[1:5], strData[5:])

def save2Txt(data, fileName):
	with open(fileName, "w") as fo:
		for line in data:
			data = line.split()
			data = list(map(float, data[1:]))
			if min(data)>=0:
				fo.write(line)
			
def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio=0.25):   # usually ratio = 0.25
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height*1.5
    roi_x2 = x2 + padding_width*0.25
    roi_y2 = y2
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2
    return roi_x1, roi_y1, roi_x2, roi_y2

def do_expand(allData):
	ret = []
	for item in allData:
		data = item.strip().split()
		image = data[0]
		bound = data[1:5]
		bound = [float(x) for x in bound]
		src = cv.imread(IMAGE_PATH+image)
		shape = src.shape
		bound = expand_roi(bound[0], bound[1], bound[2], bound[3], shape[1], shape[0])
		data[1:5] = [str(x) for x in bound]
		ret.append(" ".join(data)+"\n")
	return ret

def generate_train_test_list(rate=0.75):
	path = LABEL_PATH + "label.txt"
	fo = open(path, "r")
	allData = np.array(fo.readlines())
	fo.close()
	index = np.linspace(0, len(allData)-1, len(allData), dtype=int)
	np.random.shuffle(index)
	trainIndex = int(len(allData)*rate)
	trainData = allData[index[0:trainIndex]]
	testData = allData[index[trainIndex:]]

	trainData = do_expand(trainData)
	testData = do_expand(testData)

	save2Txt(trainData, LABEL_SPLIT_PATH+"train_ex.txt")
	save2Txt(testData, LABEL_SPLIT_PATH+"test_ex.txt")
	return trainData+testData

def load_truth(lines):
    truth = {}
    for line in lines:
        line = line.strip().split()
        name = line[0]
        if name not in truth:
            truth[name] = []
        rect = list(map(int, list(map(float, line[1:5]))))
        x = list(map(float, line[5::2]))
        y = list(map(float, line[6::2]))
        landmarks = list(zip(x, y))
        truth[name].append((rect, landmarks))
    return truth

def get_iou(rect1, rect2):
    # rect: 0-4: x1, y1, x2, y2
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    #print('wleft: ', w_left, '  hleft: ', h_left, '    wright: ', w_right, '    h_right: ', h_right)

    box1_area = width1 * height1
    box2_area = width2 * height2
    #print('inner_area: ', inner_area, '   b1: ', box1_area, '   b2: ', box2_area)
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou

def check_iou(rect1, rect2):
    # rect: 0-4: x1, y1, x2, y2
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    # print('wleft: ', w_left, '  hleft: ', h_left, '    wright: ', w_right, '    h_right: ', h_right)

    box1_area = width1 * height1
    box2_area = width2 * height2
    # print('inner_area: ', inner_area, '   b1: ', box1_area, '   b2: ', box2_area)
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou

def generate_random_crops(shape, rects, random_times):
    neg_gen_cnt = 0
    img_h = shape[0]
    img_w = shape[1]
    rect_wmin = img_w   # + 1
    rect_hmin = img_h   # + 1
    rect_wmax = 0
    rect_hmax = 0
    num_rects = len(rects)
    for rect in rects:
        w = rect[2] - rect[0] + 1
        h = rect[3] - rect[1] + 1
        if w < rect_wmin:
            rect_wmin = w
        if w > rect_wmax:
            rect_wmax = w
        if h < rect_hmin:
            rect_hmin = h
        if h > rect_hmax:
            rect_hmax = h
    random_rect_cnt = 0
    random_rects = []
    while random_rect_cnt < num_rects * random_times and neg_gen_cnt < neg_gen_thre:
        neg_gen_cnt += 1
        if img_h - rect_hmax - random_border > 0:
            top = np.random.randint(0, img_h - rect_hmax - random_border)
        else:
            top = 0
        if img_w - rect_wmax - random_border > 0:
            left = np.random.randint(0, img_w - rect_wmax - random_border)
        else:
            left = 0
        rect_wh = np.random.randint(min(rect_wmin, rect_hmin), max(rect_wmax, rect_hmax) + 1)
        rect_randw = np.random.randint(-3, 3)
        rect_randh = np.random.randint(-3, 3)
        right = left + rect_wh + rect_randw - 1
        bottom = top + rect_wh + rect_randh - 1

        good_cnt = 0
        for rect in rects:
            img_rect = [0, 0, img_w - 1, img_h - 1]
            rect_img_iou = get_iou(rect, img_rect)
            if rect_img_iou > negsample_ratio:
                random_rect_cnt += random_times
                break
            random_rect = [left, top, right, bottom]
            iou = get_iou(random_rect, rect)

            if iou < 0.2:
                good_cnt += 1
            else:
                break

        if good_cnt == num_rects:
            _iou = check_iou(random_rect, rect)
            random_rect_cnt += 1
            random_rects.append(random_rect)
    return random_rects

def genCropList():
	path = LABEL_PATH + "label.txt"
	lines = None
	with open(path, "r") as fo:
		lines = fo.readlines()
	truths = load_truth(lines)
	negtiveData = []
	for key in truths:
		rects = [x[0] for x in truths[key]]
		image = cv.imread(IMAGE_PATH+key, 0)
		random_rects = generate_random_crops(image.shape, rects, random_times)
		for rect in random_rects:
			region = image[rect[1]:rect[3], rect[0]:rect[2]]
			data = [key, str(rect[0]),str(rect[1]), str(rect[2]), str(rect[3])]
			landmarks = ["0"]*42
			data += landmarks
			negtiveData.append(" ".join(data)+"\n")
			# cv.imshow("src",region)
			# cv.waitKey(0)
	save2Txt(negtiveData, LABEL_SPLIT_PATH+"negative.txt")
	return negtiveData

def mergeNegtive2Label():
	negativeLabel = genCropList()
	nativeLabel = generate_train_test_list()

	allData = [x.strip('\n')+" 0\n" for x in negativeLabel]
	allData += [x.strip('\n')+" 1\n" for x in nativeLabel]
	allData = np.array(allData)
	
	index = np.linspace(0, len(allData)-1, len(allData), dtype=int)
	np.random.shuffle(index)
	trainIndex = int(len(allData)*0.7)

	trainData = allData[index[0:trainIndex]]
	testData = allData[index[trainIndex:]]

	def sampleWrite(fileName, ret):
		with open(LABEL_PATH+fileName, "w") as fo:
			for line in ret:
				fo.write(line)

	sampleWrite("train.txt", trainData)
	sampleWrite("test.txt", testData)

mergeNegtive2Label()
# preShow()