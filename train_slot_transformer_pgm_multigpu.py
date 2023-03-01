import os
import argparse
from slot_transformer_pgm import *
import time
import datetime
import torch.optim as optim
import torch
from PIL import Image
from torchvision.transforms import transforms
from util import log
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
from collections import OrderedDict
import torchvision.transforms.functional as TF
import random

def setup(rank, world_size):
	# initialize the process group
	dist.init_process_group("nccl", rank=rank, world_size=world_size)

class dataset(Dataset):
	def __init__(self, root_dir, dataset_type, img_size):
		self.root_dir = root_dir
		self.transforms = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
				transforms.Resize((img_size,img_size)),
			]
		)
				
		self.file_names = [root_dir + f for f in os.listdir(root_dir)
							if dataset_type in f]
			
		if dataset_type == 'train':

			
			self.train = True
		
		else:
		
			
			self.train=False


		# self.angles=[0,90,180,270] 
		# print(self.file_names[:100])
		# if dataset_type == 'train':
		# 	self.file_names = self.file_names[:10000]	
							
		self.img_size = img_size
		# self.embeddings = np.load('./embedding.npy')

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, idx):
		# data_path = os.path.join(self.root_dir, self.file_names[idx])
		data_path = self.file_names[idx]
		data = np.load(data_path)
		image = data["image"].reshape(16, 160, 160)
		target = data["target"]
		
		# if self.train:

		# 	brightness_factor =  (-1 * torch.rand(1) + 1.5).item()
		# 	if np.random.rand()<0.5:
		# 		hflip_flag=1
		# 	else:
		# 		hflip_flag=0

		# 	if np.random.rand()<0.5:
		# 		vflip_flag=1
		# 	else:
		# 		vflip_flag=0

		# 	angle = random.sample(self.angles,k=1)[0]
		# 	# angle =0
		# else:
		# 	brightness_factor =1
		# 	hflip_flag=0
		# 	vflip_flag =0
		# 	angle = 0

		
		# if vflip_flag==1 and hflip_flag==1:
		# 	img = [TF.vflip(TF.hflip(TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle))) for i in range(16)]
		# elif vflip_flag==1 and hflip_flag==0:
		# 	img = [TF.vflip(TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle)) for i in range(16)]
		# elif vflip_flag==0 and hflip_flag==1:
		# 	img = [TF.hflip(TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle)) for i in range(16)]
		# else:
		# 	img = [TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle) for i in range(16)]

		img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]
	
		# print(img[0].shape,img[1].shape)
		# resize_image = misc.imresize(data["image"][:,:,0], (self.img_size, self.img_size))
		

		
		return torch.stack(img,dim=0),target





def save_nw(slot_model,transformer_scoring_model,optimizer,epoch, name,save_path):

	
	torch.save({
		'slot_model_state_dict': slot_model.state_dict(),
		'transformer_scoring_model_state_dict': transformer_scoring_model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
	}, save_path+name)

def save_transformer_nw(transformer_scoring_model,optimizer,epoch, name):

	
	torch.save({
		
		'transformer_scoring_model_state_dict': transformer_scoring_model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
	}, '/scratch/gpfs/smondal/slot-attention-pytorch/weights/'+name)
	

def load_slot_checkpoint(slot_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path,map_location= torch.device('cpu'))


	slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	# transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return slot_model 	

	
def load_checkpoint(slot_model,transformer_scoring_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path,map_location= torch.device('cpu'))
	
# create new OrderedDict that does not contain `module.`

	new_state_dict = OrderedDict()
	for k, v in model_ckp['slot_model_state_dict'].items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	# load params
	slot_model.load_state_dict(new_state_dict)
	# print("modules>>",len(model_ckp['slot_model_state_dict']))
	# slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	new_state_dict = OrderedDict()
	for k, v in model_ckp['transformer_scoring_model_state_dict'].items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	# load params
	transformer_scoring_model.load_state_dict(new_state_dict)
	# transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return slot_model ,transformer_scoring_model	



parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_slots', default=16, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=32, type=int, help='hidden dimension size')
parser.add_argument('--depth', default=6, type=int, help='transformer number of layers')
parser.add_argument('--heads', default=8, type=int, help='transformer number of heads')
parser.add_argument('--mlp_dim', default=512, type=int, help='transformer mlp dimension')

parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=100, type=int, help='number of workers for loading data')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--run', type=str, default='1')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=1000)

parser.add_argument('--path', default='/scratch/gpfs/smondal/pgm_datasets/neutral/', type=str, help='dataset path')
parser.add_argument('--save_path', default='/scratch/gpfs/smondal/slot-attention-pytorch/weights/', type=str, help='model save path')

parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--model_name', type=str, default='slot_attention_autoencoder_transformer_scoring_multigpu')
parser.add_argument('--model_checkpoint', type=str, default='model saved checkpoint')
parser.add_argument('--apply_context_norm', type=bool, default=True)

# parser.add_argument('--accumulation_steps', type=int, default=8)

opt = parser.parse_args()
print(opt)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:" + str(opt.device) if use_cuda else "cpu")

world_size    = int(os.environ["WORLD_SIZE"])
rank          = int(os.environ["SLURM_PROCID"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
assert gpus_per_node == torch.cuda.device_count()
print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
	  f" {gpus_per_node} allocated GPUs per node.", flush=True)

dist.init_process_group("nccl", rank=rank, world_size=world_size)
if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)




local_rank = rank - gpus_per_node * (rank // gpus_per_node)
torch.cuda.set_device(local_rank)
print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")


log.info('Loading pgm datasets...')
train_data = dataset(opt.path, "train", opt.img_size)
valid_data = dataset(opt.path, "val", opt.img_size )
# test_data = dataset(opt.path, "test", opt.img_size )

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)

val_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, num_replicas=world_size, rank=rank)
val_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=opt.batch_size, sampler=val_sampler, 
											   num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)


# train_d
# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
# 						shuffle=True, num_workers=opt.num_workers)

# val_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=1,
# 						shuffle=False, num_workers=opt.num_workers)

# test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
# 						shuffle=False, num_workers=opt.num_workers)


print("Number of samples in training set>>",len(train_dataloader))
print("Number of samples in validation set>>",len(val_dataloader))

log.info('Building model...')
slot_model = SlotAttentionAutoEncoder((opt.img_size,opt.img_size), opt.num_slots, opt.num_iterations, opt.hid_dim)
# slot_model = load_slot_checkpoint(slot_model,opt.model_checkpoint)
# slot_model = slot_model.to(local_rank)
# for param in slot_model.parameters():
# 	param.requires_grad = False

transformer_scoring_model = scoring_model(opt,opt.hid_dim,opt.depth,opt.heads,opt.mlp_dim,opt.num_slots)
# slot_model, transformer_scoring_model = load_checkpoint(slot_model,transformer_scoring_model,opt.model_checkpoint)

ddp_slot_model = DDP(slot_model.to(local_rank), device_ids=[local_rank])
# transformer_scoring_model = scoring_model(opt,opt.hid_dim,opt.depth,opt.heads,opt.mlp_dim,opt.num_slots).to(local_rank)
ddp_transformer_scoring_model = DDP(transformer_scoring_model.to(local_rank), device_ids=[local_rank])

# slot_model = load_checkpoint(slot_model,opt.model_checkpoint)
# slot_model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss()

params = [{'params': list(ddp_slot_model.parameters()) + list(ddp_transformer_scoring_model.parameters())}]
# params = [{'params': ddp_transformer_scoring_model.parameters()}]

log.info('Setting up optimizer...')

optimizer = optim.Adam(params, lr=opt.learning_rate)

log.info('Training begins...')
start = time.time()
i = 0
# optimizer.zero_grad()
max_val_acc=0
for epoch in range(opt.num_epochs):
	train_sampler.set_epoch(epoch)
	ddp_slot_model.train()
	# slot_model.eval()
	
	ddp_transformer_scoring_model.train()

	
	all_trainmseloss = []
	all_trainceloss=[]
	all_trainacc = []
	all_valacc=[]
   
	# total_train_images = torch.Tensor().to(device).float()
	# total_train_recons_combined = torch.Tensor().to(device).float()
	# total_train_recons = torch.Tensor().to(device).float()
	# total_train_masks = torch.Tensor().to(device).float()
	
	for batch_idx, (img,target) in enumerate(train_dataloader):
		# print("image and target shape>>",img.shape,target.shape)
# 		# print(torch.max(img), torch.min(img),img.shape)
		i += 1

		if i < opt.warmup_steps:
			learning_rate = opt.learning_rate * (i / opt.warmup_steps)
		else:
			learning_rate = opt.learning_rate

		# learning_rate = learning_rate * (opt.decay_rate ** (
		# 	i / opt.decay_steps))

		optimizer.param_groups[0]['lr'] = learning_rate
		img = img.to(local_rank) #.unsqueeze(1).float()
		target = target.to(local_rank)
		slots_seq =[]
		recon_combined_seq =[]
		recons_seq=[]
		masks_seq=[]
		for idx in range(img.shape[1]):
			# print(idx)

		
			recon_combined, recons, masks, slots,_ = ddp_slot_model(img[:,idx],local_rank)
			# recon_combined, recons, masks, slots,_ = slot_model(img[:,idx],local_rank)
			
			# slots = ddp_slot_model(img[:,idx],local_rank)
			
			slots_seq.append(slots)
			recon_combined_seq.append(recon_combined)
			recons_seq.append(recons)
			masks_seq.append(masks)
			del recon_combined,recons, masks, slots

		given_panels = torch.stack(slots_seq,dim=1)[:,:8]
		answer_panels = torch.stack(slots_seq,dim=1)[:,8:]
		# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

		scores = ddp_transformer_scoring_model(given_panels,answer_panels,local_rank)
		# print("scores and target>>",scores,target)
		pred = scores.argmax(1)
		acc = torch.eq(pred,target).float().mean().item() * 100.0
# 		# print(torch.max(recon_combined), torch.min(recon_combined))
# 		# print(img.shape, recon_combined.shape, recons.shape, masks.shape, slots.shape)
		# if batch_idx<10:
		# 	total_train_images = torch.cat((total_train_images,img),dim=0)
		# 	total_train_recons_combined = torch.cat((total_train_recons_combined,torch.stack(recon_combined_seq,dim=1)),dim=0)
		# 	total_train_recons = torch.cat((total_train_recons,torch.stack(recons_seq,dim=1)),dim=0)
		# 	total_train_masks = torch.cat((total_train_masks,torch.stack(masks_seq,dim=1)),dim=0)
			
		
		# print("mse loss>>>",mse_criterion(torch.stack(recon_combined_seq,dim=1).squeeze(4), img))
		# print("ce loss>>",ce_criterion(scores,target))
		# print("recon combined seq shape>>",torch.stack(recon_combined_seq,dim=1).shape)
		loss = 1000*mse_criterion(torch.stack(recon_combined_seq,dim=1), img) + ce_criterion(scores,target)
		# loss = ce_criterion(scores,target)

		all_trainmseloss.append(mse_criterion(torch.stack(recon_combined_seq,dim=1), img).item())
		all_trainceloss.append(ce_criterion(scores,target).item())
		
		all_trainacc.append(acc)
# 		del recons, masks, slots
		# loss = loss / opt.accumulation_steps   
		optimizer.zero_grad()
		loss.backward()
	

		optimizer.step()
		
		# print("learning rate>>>",learning_rate)
		# for j, para in enumerate(transformer_scoring_model.parameters()):
		#     print(f'{j + 1}th parameter tensor:', para.shape)
		#     print(para)
		#     print(para.grad)
		# if (batch_idx+1) % opt.accumulation_steps == 0:             # Wait for several backward steps
		# 	optimizer.step()                            # Now we can do an optimizer step
		# 	optimizer.zero_grad()  
		
		if batch_idx % opt.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_dataloader)) + '] ' + \
					 '[Total Loss = ' + '{:.4f}'.format(loss.item()) + '] ' +\
					 '[MSE Loss = ' + '{:.4f}'.format(mse_criterion(torch.stack(recon_combined_seq,dim=1), img).item()) + '] ' +\
					 '[CE Loss = ' + '{:.4f}'.format(ce_criterion(scores,target).item()) + '] ' +\
					 '[Accuracy = ' + '{:.2f}'.format(acc) + ']'

					  )

		# if batch_idx % opt.save_interval ==0:
		# 	np.savez('predictions/{}_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_extrapolation_train_images_recons_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy() )

		# 	save_nw(ddp_slot_model,ddp_transformer_scoring_model,optimizer,epoch,'{}_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_wholetrainset_extrapolation_run_{}_best.pth.tar'.format(opt.model_name,opt.run))
		# del img,recon_combined_seq,slots_seq,given_panels,answer_panels
	print("Average training reconstruction loss>>",np.mean(np.array(all_trainmseloss)))
	print("Average training cross entropy loss>>",np.mean(np.array(all_trainceloss)))
	print("Average training accuracy>>",np.mean(np.array(all_trainacc)))
	# torch.cuda.empty_cache()


	ddp_slot_model.eval()
	# slot_model.eval()
	
	ddp_transformer_scoring_model.eval()

	for batch_idx, (img,target) in enumerate(val_dataloader):
		# print("image and target shape>>",img.shape,target.shape)
# 		# print(torch.max(img), torch.min(img),img.shape)
		
		img = img.to(local_rank) #.unsqueeze(1).float()
		target = target.to(local_rank)
		slots_seq =[]
		recon_combined_seq =[]
		recons_seq=[]
		masks_seq=[]
		for idx in range(img.shape[1]):
			# print(idx)

		
			recon_combined, recons, masks, slots,_ = ddp_slot_model(img[:,idx],local_rank)
			# recon_combined, recons, masks, slots,_ = slot_model(img[:,idx],local_rank)
			
			# slots = ddp_slot_model(img[:,idx],local_rank)
			
			slots_seq.append(slots)
			recon_combined_seq.append(recon_combined)
			recons_seq.append(recons)
			masks_seq.append(masks)
			del recon_combined,recons, masks, slots


		given_panels = torch.stack(slots_seq,dim=1)[:,:8]
		answer_panels = torch.stack(slots_seq,dim=1)[:,8:]
		# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

		scores = ddp_transformer_scoring_model(given_panels,answer_panels,local_rank)
		# print("scores and target>>",scores,target)
		pred = scores.argmax(1)
		acc = torch.eq(pred,target).float().mean().item() * 100.0
# 		# print(torch.max(recon_combined), torch.min(recon_combined))
# 		# print(img.shape, recon_combined.shape, recons.shape, masks.shape, slots.shape)
		# if batch_idx<10:
		# 	total_train_images = torch.cat((total_train_images,img),dim=0)
		# 	total_train_recons_combined = torch.cat((total_train_recons_combined,torch.stack(recon_combined_seq,dim=1)),dim=0)
		# 	total_train_recons = torch.cat((total_train_recons,torch.stack(recons_seq,dim=1)),dim=0)
		# 	total_train_masks = torch.cat((total_train_masks,torch.stack(masks_seq,dim=1)),dim=0)
			
		
		all_valacc.append(acc)
# 		del recons, masks, slots
		# loss = loss / opt.accumulation_steps   
		

		# if batch_idx % opt.save_interval ==0:
		# 	np.savez('predictions/{}_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_extrapolation_train_images_recons_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy() )

		# 	save_nw(ddp_slot_model,ddp_transformer_scoring_model,optimizer,epoch,'{}_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_wholetrainset_extrapolation_run_{}_best.pth.tar'.format(opt.model_name,opt.run))

	print("Average validation accuracy>>",np.mean(np.array(all_valacc)))
	if np.mean(np.array(all_valacc)) > max_val_acc:
		print("Validation accuracy increased from %s to %s"%(max_val_acc,np.mean(np.array(all_valacc))))
		max_val_acc = np.mean(np.array(all_valacc))
		print("Saving model$$$$")
		save_nw(ddp_slot_model,ddp_transformer_scoring_model,optimizer,epoch,'{}_1000weightmse_tcn_16slots_dspritesdecoder_warmup_lowerlr_nolrdecay_morelayers_rowcolposemb_pgm_wholetrainset_neutral_run_{}_best.pth.tar'.format(opt.model_name,opt.run),opt.save_path)
		# save_transformer_nw(ddp_transformer_scoring_model,optimizer,epoch,'{}_tcn_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_wholetrainset_extrapolation_run_{}_best.pth.tar'.format(opt.model_name,opt.run))
		
		# save_transformer_nw(ddp_transformer_scoring_model,optimizer,epoch,'{}_tcn_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_wholetrainset_neutral_run_{}_best.pth.tar'.format(opt.model_name,opt.run))
	
	else:
		print("Validation accuracy didn't increase and hence skipping model saving!!!!")
	print("Best validation accuracy till now >>",max_val_acc)
	
dist.destroy_process_group()	

# 	print("Average training reconstruction loss>>",np.mean(np.array(all_trainloss)))
# 	np.savez('predictions/{}_15slots_pgm_neutral_train_images_recons_masks.npz'.format(opt.model_name), images= total_train_images.cpu().detach().numpy() ,recon_combined = total_train_recons_combined.cpu().detach().numpy(),recons = total_train_recons.cpu().detach().numpy(),masks= total_train_masks.cpu().detach().numpy() )
		

