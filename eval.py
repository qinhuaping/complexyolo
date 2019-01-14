import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
from scipy import misc
from region_loss import RegionLoss

from utils import *


def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
	cv2.line(img, pt1, pt2, color, lineWidth)
	cv2.line(img, pt2, pt3, color, lineWidth)
	cv2.line(img, pt3, pt4, color, lineWidth)
	cv2.line(img, pt1, pt4, color, lineWidth)


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
	anchor_step = int(len(anchors)/num_anchors)

	if output.dim() == 3:
		output = output.unsqueeze(0)
	batch = output.size(0)
	print("batch is ",batch)
	assert(output.size(1) == (7+num_classes)*num_anchors)
	h = output.size(2)  # 16
	w = output.size(3)  # 32
	print("h w",h,w)
	nB = output.data.size(0)
	nA = num_anchors     # num_anchors = 5
	nC = num_classes     # num_classes = 8
	nH = output.data.size(2)  # nH  16
	nW = output.data.size(3)  # nW  32
	anchor_step = int(len(anchors)/num_anchors)

	output = output.view(nB, nA, (7+nC), nH, nW)

	x = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
	y = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
	w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
	l = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
	im = output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)
	re = output.index_select(2, Variable(torch.cuda.LongTensor([5]))).view(nB, nA, nH, nW)
	conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([6]))).view(nB, nA, nH, nW))
	cls = output.index_select(2, Variable(torch.linspace(7, 7+nC-1, nC).long().cuda()))
	cls = cls.view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(nB*nA*nH*nW, nC)

	pred_boxes = torch.cuda.FloatTensor(7, nB*nA*nH*nW)
	grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
	grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
	anchor_w = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([0])).cuda()
	anchor_l = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([1])).cuda()
	anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
	anchor_l = anchor_l.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

	pred_boxes[0] = x.data.view(nB*nA*nH*nW).cuda() + grid_x
	pred_boxes[1] = y.data.view(nB*nA*nH*nW).cuda() + grid_y
	pred_boxes[2] = torch.exp(w.data).view(nB*nA*nH*nW).cuda() * anchor_w
	pred_boxes[3] = torch.exp(l.data).view(nB*nA*nH*nW).cuda() * anchor_l
	# pred_boxes[4] = np.arctan2(im,re).data.view(nB*nA*nH*nW).cuda()
	pred_boxes[4] = im.data.view(nB*nA*nH*nW).cuda()
	pred_boxes[5] = re.data.view(nB*nA*nH*nW).cuda()

	pred_boxes[6] = conf.data.view(nB*nA*nH*nW).cuda()
	#pred_boxes[7:(7+nC)] = cls.data.view(nC, nB*nA*nH*nW).cuda()
	pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 7))  # torch.Size([2560, 15])

	all_boxes = []
	for i in range(2560):
		#print(pred_boxes[i][6])
		if pred_boxes[i][6]>conf_thresh:
			all_boxes.append(pred_boxes[i])
	return all_boxes


# classes
# class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram' ]


bc = {}
bc['minX'] = 0; bc['maxX'] = 80; bc['minY'] = -40; bc['maxY'] = 40
bc['minZ'] = -2; bc['maxZ'] = 1.25

#torch.cuda.set_device(0)
#rgb_map = rgb_map.view(rgb_map.data.size(0), rgb_map.data.size(3), rgb_map.data.size(1), rgb_map.data.size(2))
model = torch.load('ComplexYOLO_epoch390')
model.cuda()
region_loss = RegionLoss(num_classes=8, num_anchors=5)
for file_i in range(100):
	test_i = str(file_i).zfill(6)

	lidar_file = '/home/Kitti/object/training/velodyne/'+test_i+'.bin'
	calib_file = '/home/Kitti/object/training/calib/'+test_i+'.txt'
	label_file = '/home/Kitti/object/training/label_2/'+test_i+'.txt'
	image_file = '/home/Kitti/object/training/image_2/'+test_i+'.png'
	# load target data
	calib = load_kitti_calib(calib_file)
	target = get_target(label_file, calib['Tr_velo2cam'])
	#print(target)
	print(test_i)
	# load point cloud data
	a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
	b = removePoints(a,bc)
	#print(b.shape)
	rgb_map = makeBVFeature(b, bc ,(float)(40)/512)

	misc.imsave('eval_bv.png', rgb_map)
	img = cv2.imread('eval_bv.png')
	image = cv2.imread(image_file)
	for j in range(50):
		if target[j][1] == 0:
			break
		img_y = int(target[j][1] * 1024.0)  # 32 cell = 1024 pixels
		img_x = int(target[j][2] * 512.0)  # 16 cell = 512 pixels
		img_width = int(target[j][3] * 1024.0)  # 32 cell = 1024 pixels
		img_height = int(target[j][4] * 512.0)  # 16 cell = 512 pixels
		sin_value = float(target[j][5])
		cos_value = float(target[j][6])
		#print(cos_value,sin_value)
		T = np.array([[cos_value,sin_value],
					 [-sin_value,cos_value]])
		#print("T::")
		#print(T)

		rect_top1 = int(img_y - img_width / 2)
		rect_top2 = int(img_x - img_height / 2)
		rect_bottom1 = int(img_y + img_width / 2)
		rect_bottom2 = int(img_x + img_height / 2)


		Pose = np.array([[- img_width / 2,- img_height / 2],
						 [img_width / 2,img_height / 2]])

		Pose_T = np.transpose(Pose)

		Pose1 = np.array([[- img_width / 2, img_height / 2],
						 [img_width / 2,-img_height / 2]])
		Pose_T1 = np.transpose(Pose1)
		new_pos = np.dot(T,Pose_T)
		new_pos1 = np.dot(T,Pose_T1)
		pos1 = new_pos[:,0]
		pos2 = new_pos[:,1]
		pos3 = new_pos1[:,0]
		pos4 = new_pos1[:,1]
		#print(rect_top1,rect_top2,rect_bottom1,rect_bottom2)
		#cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (0, 0, 255), 1)
		#print(int(pos1[0])+img_y, int(pos1[1]+img_x),int(pos2[0]+img_y), int(pos2[1]+img_x))
		cv2.line(img, (int(pos1[0])+img_y, int(pos1[1])+img_x), (int(pos3[0]+img_y), int(pos3[1]+img_x)), (0, 255, 255), 1)
		cv2.line(img, (int(pos3[0] + img_y), int(pos3[1] + img_x)), (int(pos2[0] + img_y), int(pos2[1] + img_x)),(0, 255, 255), 1)
		cv2.line(img, (int(pos2[0] + img_y), int(pos2[1] + img_x)), (int(pos4[0] + img_y), int(pos4[1] + img_x)),(0, 255, 255), 1)
		cv2.line(img, (int(pos4[0] + img_y), int(pos4[1] + img_x)), (int(pos1[0] + img_y), int(pos1[1] + img_x)),(0, 255, 255), 1)
		cv2.namedWindow('showimage')

		cv2.imshow("originimahe",image)

#		misc.imsave('eval_bv'+test_i+'.png',img)


	# load trained model  and  forward
	input = torch.from_numpy(rgb_map)       # (512, 1024, 3)
	input = input.reshape(1,3,512,1024)

	output = model(input.float().cuda())    #torch.Size([1, 75, 16, 32])

	# eval result
	conf_thresh   = 0.4
	nms_thresh    = 0.4
	num_classes = int(8)
	num_anchors = int(5)
	#img = cv2.imread('eval_bv.png')

	print(output.shape)
	#loss = region_loss(output, target)

	all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
	print(len(all_boxes))

	for i in range(len(all_boxes)):

		pred_img_y = int(all_boxes[i][0]*1024.0/32.0)   # 32 cell = 1024 pixels
		pred_img_x = int(all_boxes[i][1]*512.0/16.0)    # 16 cell = 512 pixels
		pred_img_width  = int(all_boxes[i][2]*1024.0/32.0)   # 32 cell = 1024 pixels
		pred_img_height = int(all_boxes[i][3]*512.0/16.0)
		print(pred_img_y)
		sin_value = float(all_boxes[i][4])
		cos_value = float(all_boxes[i][5])
		# print(cos_value,sin_value)
		T = np.array([[cos_value, sin_value],[-sin_value, cos_value]])
		# print("T::")
		# print(T)

		rect_top1 = int(pred_img_y - pred_img_width / 2)
		rect_top2 = int(pred_img_x - pred_img_height / 2)
		rect_bottom1 = int(pred_img_y + pred_img_width / 2)
		rect_bottom2 = int(pred_img_x + pred_img_height / 2)

		Pose = np.array([[- pred_img_width / 2, - pred_img_height / 2],[pred_img_width / 2, pred_img_height / 2]])

		Pose_T = np.transpose(Pose)

		Pose1 = np.array([[- pred_img_width / 2, pred_img_height / 2],[pred_img_width / 2, -pred_img_height / 2]])
		Pose_T1 = np.transpose(Pose1)
		new_pos = np.dot(T, Pose_T)
		new_pos1 = np.dot(T, Pose_T1)
		pos1 = new_pos[:, 0]
		pos2 = new_pos[:, 1]
		pos3 = new_pos1[:, 0]
		pos4 = new_pos1[:, 1]

		cv2.line(img, (int(pos1[0]) + pred_img_y, int(pos1[1]) + pred_img_x), (int(pos3[0] + pred_img_y), int(pos3[1] + pred_img_x)),(0, 255, 0), 1)
		cv2.line(img, (int(pos3[0] + pred_img_y), int(pos3[1] + pred_img_x)), (int(pos2[0] + pred_img_y), int(pos2[1] + pred_img_x)),(0, 255, 0), 1)
		cv2.line(img, (int(pos2[0] + pred_img_y), int(pos2[1] + pred_img_x)), (int(pos4[0] + pred_img_y), int(pos4[1] + pred_img_x)),(0, 255, 0), 1)
		cv2.line(img, (int(pos4[0] + pred_img_y), int(pos4[1] + pred_img_x)), (int(pos1[0] + pred_img_y), int(pos1[1] + pred_img_x)),(0, 255, 0), 1)
		cv2.imshow("showimage", img)
		cv2.waitKey(100)
	misc.imsave('eval_bv'+test_i+'.png',img)

