import torch 
import torch.nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from pprintpp import pprint
from torch vision.transforms import Compose, ToTensor, Resize
from tqdm.autonotebook import tqdm
from torchmetrics.detection import MeanAveragePrecision

class VOCDataset(VOCDetection):
	def __init__(self, root, year, image_set, download, transform):
		super().__init__(root, year, image_set, download, transform)
		self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'board', 'bottle', 'bus', 'car', 'cat',
						'cow', 'dining table', 'dog', 'horse', 'motobike', 'person', 'pottedplant', 'sheep',
						'train', 'tvmonitor']

	def __getitem__(self, item):
		image, ori_target = super().__getitem__(item)
		boxes = []
		labels = []
		for obj in ori_target["annotation"]["object"]:
			bbox = obj["bndbox"]
			xmin = int(bbox["xmin"])
			ymin = int(bbox["ymin"])
			xmax = int(bbox["xmax"])
			ymax = int(bbox["ymax"])
			label = obj["name"]
			boxes.append([xmin, ymin, xmax, ymax])
			labels.append(self.classes.index(label))
		final_target = {"boxes": torch.FloatTensor(boxes), "labels": torch.LongTensor(labels)}
		return image, final_target

	def collate_fn(batch):
		print(batch)

	def train():
		epochs = 10
		batch_size = 4
		lr = 1e-3
		device = torch.device("cuda" if torch.cuda.is_avaiable() else "cpu")
		transform = Compose([
			ToTensor(),
		])
		train_dataset = VOCDataset("data", year = "2012", image_set="train", download = False, transform = transform)
		train_dataloader = DataLoader(
			dataset = train_dataset,
			batch_size = batch_size,
			num_workers = 4,
			shuffer = True,
			drop_last = True,
			collate_fn = collate_fn()
		)
		val_dataset = VOCDataset(root:"data", year="2012", image_set="val", download = False, transform = transform)
		val_dataloader = DataLoader(
			dataset = val_dataset,
			batch_size = batch_size,
			num_workers = 4,
			shuffer = False,
			drop_last = False,
		)

		# model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT) 
		model = fasterrcnn_resnet50_fpn(weights=fasterrcnn_mobilenet_v3_large_320_FPN_Weights.DEFAULT, trainable_backbone_layers=1)
		in_feature = model.roi.heads.box_predictor.cls_score.in_features
		model.roi.heads.box_predictor.cls_score == nn.Linear(in_feature=in_features, out_features = num_classes)
		model.roi.heads.box_predictor.bbox_pred == nn.Linear(in_feature=in_features, out_features = num_classes * 4)
		model.to(device)
		model.train()
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9)
		for epochs in range(epochs):
			# model.train()
			# progress_bar = tqdm(train_dataloader, colour = "cyan")
			# for images, targets in  train_dataloader:
				# images = [image.to(device) for image in images]
						
				# # advance way
				# # targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

				# # basic way
				# target_list = []
				# for target in targets:
				# 	target_list.append({
						# "boxes": target["boxes"].to(device),
						# "labels": target["labels"].to(device)
				# 		})

				# losses = model(images.to(device), targets.to(device))
				# total_loss = sum([loss for loss in losses.value()])
				# progress_bar.set_description("Epoch {}/{}, Loss: {:0.4f}".format(epoch, epochs, total_loss))
				
				# optimizer.zero_grad()		#lam sach buffer
				# total_loss.backward()		#Tinh gradient
				# optimizer.step()


			# bộ val cũng như trên

			model.eval()
			progress_bar = tqdm(val_dataloader, colour = "yellow")
			metric = MeanAveragePrecision(iou_type = "bbox")
			for images, targets in progress_bar:
				images = [image.to(device) for image in images]
						
				
				predictions = model(images)
				
				cpu_prediction = []
				for predictions.append({
						"boxes": prediction["boxes"].to("cpu"),
						"labels": prediction["labels"].to("cpu"),
						"scores": prediction["scores"].to("cpu"),						
					})

				metric.update(predictions.to(torch.device("cpu")), targets)
			pprint(metric.compute())

		# Đánh giá mô hình: càng gần 1 thì càng tốt, càng gần 0 càng tồi
		# Pytorch có library torch metrics, giúp đánh giá mAP (mean average precision)

if __name__ == '__main__':
	train()
