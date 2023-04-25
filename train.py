import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np
import torch.backends.cuda as tbc


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()

	# if weight_pt is not None:
	# 	model.load_state_dict(torch.load(weight_pt))

	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1) # giam lr theo thoi gian
	tbc.split_kernel_size = 64

	for epoch in range(epoch_iter):	
		model.train()
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path,'model_epoch_{}.pth'.format(epoch+1)))

import argparse
import uuid
def opt_parse():
	parser = argparse.ArgumentParser(description= " param for detect")
	parser.add_argument("--train-img-fol",type=str,required=True,help="path of folder image train")
	parser.add_argument("--train-lb-fol",type=str,required=True,help="path of label folder")
	parser.add_argument("--save-fol",type=str,required=True,help="path of output weight")
	# parser.add_argument("--weight-pt",str=str,default=None,help="optional about input pre-trained, none is train from scratch")
	parser.add_argument("--batch",type=int,default=16,help="batch size of train data")
	parser.add_argument("--lr",type=float,default=1e-3,help="start learning rate")
	parser.add_argument("--epoch",type=int,default=10,help="epoch train")
	parser.add_argument("--check-point",type=int,default=5,help="each check-point time will save weight")
	return parser.parse_args()

if __name__ == '__main__':
	args = opt_parse()
	train_img_path = args.train_img_fol
	train_gt_path  = args.train_lb_fol
	pths_path      = args.save_fol
	batch_size     = args.batch
	lr             = args.lr
	num_workers    = 1
	epoch_iter     = args.epoch
	save_interval  = args.check_point
	# weight	= args.weight_pt
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)	
	
