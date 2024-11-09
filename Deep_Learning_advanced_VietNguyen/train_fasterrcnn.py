import torch 
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from pprintpp import pprint
from torch vision.transforms import Compose, ToTensor, Resize


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
		batch_size = 4
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

		model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT) 
		model.train()
		for images, targets in train_dataloader:
			losses = model(images, targets)
			print(loses)

if __name__ == '__main__':
	train()
