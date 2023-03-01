import os
import argparse
from slot_transformer_v2 import *
import time
import datetime
import torch.optim as optim
import torch
from PIL import Image
from torchvision.transforms import transforms
from util import log
from torch.utils.data import Dataset, DataLoader
import glob
import torchvision.transforms.functional as TF
import random
figure_configuration_names = ['center_single', 'distribute_four', 'distribute_nine', 'in_center_single_out_center_single', 'in_distribute_four_out_center_single', 'left_center_single_right_center_single', 'up_center_single_down_center_single']

class ToTensor(object):
	def __call__(self, sample):
		return torch.tensor(sample, dtype=torch.float32)

class RAVENdataset(Dataset):
	def __init__(self,  root_dir, dataset_type, figure_configurations, img_size,N_slots, shuffle=False):
		self.root_dir = root_dir
		self.transforms = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Resize((img_size,img_size)),
					
					


				]
			)
		if dataset_type == 'train':

			
			self.train = True
		
		else:
		
			
			self.train=False


		self.file_names = []
		self.angles=[0,90,180,270]
		for idx in figure_configurations:
			tmp = [f for f in glob.glob(os.path.join(root_dir, figure_configuration_names[idx], "*.npz")) if dataset_type in os.path.basename(f)]

			self.file_names += tmp
		self.img_size = img_size
		self.N_slots = N_slots   
		self.shuffle = shuffle
		self.switch = [3,4,5,0,1,2,6,7]     


			 
		# print(self.file_names[:100])
		# if dataset_type == 'train':
		# 	self.file_names = self.file_names[:10000]	
		
		# self.embeddings = np.load('./embedding.npy')

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, idx):
		# data_path = os.path.join(self.root_dir, self.file_names[idx])

		data_path = self.file_names[idx]
		data = np.load(data_path)
		image = data["image"].reshape(16, 160, 160)
		target = data["targ"]
		masks = data["mask"].reshape(16,160,160)

		if self.shuffle:
			context = image[:8, :, :]
			choices = image[8:, :, :]
			masks_context = masks[:8,:,:]
			masks_choices = masks[8:,:,:]

			indices = np.arange(8)
			np.random.shuffle(indices)
			new_target = np.where(indices == target)[0][0]
			new_choices = choices[indices, :, :]
			new_masks_choices = masks_choices[indices,:,:]

			switch_2_rows = np.random.rand()            
			if switch_2_rows < 0.5:                
				context = context[self.switch, :, :]
				masks_context = masks_context[self.switch,:,:]

			image = np.concatenate((context, new_choices))
			masks = np.concatenate((masks_context,new_masks_choices))
			target = new_target

		# resize_image = []
		# for idx in range(0, 16):
		if self.train:

			brightness_factor =   (-1 * torch.rand(1) + 1.5).item()
			if np.random.rand()<0.5:
				hflip_flag=1
			else:
				hflip_flag=0

			if np.random.rand()<0.5:
				vflip_flag=1
			else:
				vflip_flag=0

			angle = random.sample(self.angles,k=1)[0]
			# angle =0
		else:
			brightness_factor =1
			hflip_flag=0
			vflip_flag =0
			angle = 0
		
		if vflip_flag==1 and hflip_flag==1:
			img = [TF.vflip(TF.hflip(TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle))) for i in range(16)]
		elif vflip_flag==1 and hflip_flag==0:
			img = [TF.vflip(TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle)) for i in range(16)]
		elif vflip_flag==0 and hflip_flag==1:
			img = [TF.hflip(TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle)) for i in range(16)]
		else:
			img = [TF.rotate(TF.adjust_brightness(self.transforms(Image.fromarray(image[i].astype(np.uint8))),brightness_factor),angle=angle) for i in range(16)]
		# img = [self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)]
	
			# resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
		resize_image = torch.stack(img,dim=0)
		
		
		del data
		
		return resize_image, target

		
		# print(img[0].shape,img[1].shape)
		# resize_image = misc.imresize(data["image"][:,:,0], (self.img_size, self.img_size))
		

		
		





def save_nw(slot_model,transformer_scoring_model,optimizer,epoch, name,save_path):

	
	torch.save({
		'slot_model_state_dict': slot_model.state_dict(),
		'transformer_scoring_model_state_dict': transformer_scoring_model.state_dict(),
		
		
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
	}, save_path+name)
	

def load_slot_checkpoint(slot_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
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
	model_ckp = torch.load(checkpoint_path)
	slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	transformer_scoring_model.load_state_dict(model_ckp['transformer_scoring_model_state_dict'])
	
	return slot_model ,transformer_scoring_model	



parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=9, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=32, type=int, help='hidden dimension size')
parser.add_argument('--depth', default=6, type=int, help='transformer number of layers')
parser.add_argument('--heads', default=8, type=int, help='transformer number of heads')
parser.add_argument('--mlp_dim', default=512, type=int, help='transformer mlp dimension')

parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=75000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=300, type=int, help='number of workers for loading data')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--run', type=str, default='1')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=1000)

parser.add_argument('--path', default='/scratch/gpfs/smondal/I-RAVEN/', type=str, help='dataset path')
parser.add_argument('--save_path', default='/scratch/gpfs/smondal/slot-attention-pytorch/weights/', type=str, help='model save path')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--model_name', type=str, default='slot_attention_autoencoder_transformer_scoring')
parser.add_argument('--model_checkpoint', type=str, default='model saved checkpoint')
parser.add_argument('--apply_context_norm', type=bool, default=True)

# parser.add_argument('--accumulation_steps', type=int, default=8)

opt = parser.parse_args()
print(opt)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:" + str(opt.device) if use_cuda else "cpu")


log.info('Loading RAVEN datasets...')
configurations = [0,1,2,3,4,5,6]
# configurations=[3]
train_data = RAVENdataset(opt.path, "train",configurations, opt.img_size,opt.num_slots,shuffle = True)
valid_data = RAVENdataset(opt.path, "val", configurations, opt.img_size,opt.num_slots)
test_data = RAVENdataset(opt.path, "test", configurations, opt.img_size,opt.num_slots)


train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
						shuffle=True, num_workers=opt.num_workers)

val_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=opt.batch_size,
						shuffle=False, num_workers=opt.num_workers)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
						shuffle=False, num_workers=opt.num_workers)


print("Number of samples in training set>>",len(train_dataloader))
print("Number of samples in validation set>>",len(val_dataloader))

print("Number of samples in test set>>",len(test_dataloader))

log.info('Building model...')
# slotmask_model = SlotMaskAttention((opt.img_size,opt.img_size), opt.num_slots, opt.hid_dim).to(device)
slot_model = SlotAttentionAutoEncoder((opt.img_size,opt.img_size), opt.num_slots,opt.num_iterations, opt.hid_dim).to(device)

transformer_scoring_model = scoring_model(opt,opt.hid_dim,opt.depth,opt.heads,opt.mlp_dim,opt.num_slots).to(device)
# slot_model = load_slot_checkpoint(slot_model,opt.model_checkpoint)
# slot_model, transformer_scoring_model = load_checkpoint(slot_model,transformer_scoring_model,opt.model_checkpoint)
# slot_model,transformer_scoring_model = load_checkpoint(slot_model,transformer_scoring_model,'weights/slot_attention_augmentations_pretrained_frozen_autoencoder_transformer_scoring_tcn_shuffling_augmentations_9slots_nowarmup_nolrdecay_rowcolposemb_raven_allconfigs_run_1_best.pth.tar')

# slot_model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])
mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss()

params = [{'params': list(slot_model.parameters()) + list(transformer_scoring_model.parameters())}]
# params = [{'params': transformer_scoring_model.parameters()}]
model_parameters = filter(lambda p: p.requires_grad, list(slot_model.parameters()) + list(transformer_scoring_model.parameters()))
# model_parameters = filter(lambda p: p.requires_grad, slot_model.parameters())

print("trainable parameters>>",sum([np.prod(p.size()) for p in model_parameters]))

# params = [{'params': slot_model.parameters()}]

log.info('Setting up optimizer...')

optimizer = optim.Adam(params, lr=opt.learning_rate)

log.info('Training begins...')
start = time.time()
i = 0
max_val_acc=0
# optimizer.zero_grad()
for epoch in range(opt.num_epochs):
	slot_model.train()
	transformer_scoring_model.train()

	
	# all_trainloss = []
	all_trainmseloss = []
	all_trainceloss=[]
	all_trainacc = []
   
	# total_train_images = torch.Tensor().to(device).float()
	# total_train_recons_combined = torch.Tensor().to(device).float()
	# total_train_recons = torch.Tensor().to(device).float()
	# total_train_masks = torch.Tensor().to(device).float()
	
	for batch_idx, (img,target) in enumerate(train_dataloader):
		# print("image and masks and target shape>>",img.shape,masks.shape,target.shape)
		# print(torch.max(img), torch.min(img),torch.max(masks),torch.min(masks))
		i += 1

		if i < opt.warmup_steps:
			learning_rate = opt.learning_rate * (i / opt.warmup_steps)
		else:
			learning_rate = opt.learning_rate

		# learning_rate = learning_rate * (opt.decay_rate ** (
		# 	i / opt.decay_steps))

		optimizer.param_groups[0]['lr'] = learning_rate
		img = img.to(device).float() #.unsqueeze(1).float()
		# masks = masks.to(device).float()
		target = target.to(device)
		slots_seq =[]
		recon_combined_seq =[]
		recons_seq=[]
		masks_seq=[]
		attn_seq=[]
		for idx in range(img.shape[1]):
			# print(idx)
			# print(img[:,idx].shape,masks[:,idx].shape)
		
			# slots = slotmask_model(img[:,idx],masks[:,idx],device)
			recon_combined, recons, masks, slots,attn = slot_model(img[:,idx],device)
			# print("mask max and min>>", torch.max(masks), torch.min(masks))
			slots_seq.append(slots)
			recon_combined_seq.append(recon_combined)
			recons_seq.append(recons)
			masks_seq.append(masks)
			attn_seq.append(attn)
			del recon_combined,recons, masks, slots,attn

		

			
		

		given_panels = torch.stack(slots_seq,dim=1)[:,:8]
		answer_panels = torch.stack(slots_seq,dim=1)[:,8:]
		# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

		scores = transformer_scoring_model(given_panels,answer_panels,device)
		# print(scores.shape)
		# print("scores and target>>",scores,target)
		pred = scores.argmax(1)
		acc = torch.eq(pred,target).float().mean().item() * 100.0
		all_trainacc.append(acc)
# 		
# print(torch.max(recon_combined), torch.min(recon_combined))
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
		
		# loss =  ce_criterion(scores,target)
		# loss = ce_criterion(scores,target)
		
		all_trainmseloss.append(mse_criterion(torch.stack(recon_combined_seq,dim=1), img).item())
		all_trainceloss.append(ce_criterion(scores,target).item())
		
# 		all_trainloss.append(loss.item())

# 		del recons, masks, slots
		# loss = loss / opt.accumulation_steps   
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# print("learning rate>>>",learning_rate)
		# for j, para in enumerate(slotmask_model.parameters()):
		#     print(f'{j + 1}th parameter tensor:', para.shape)
		#     # print(para)
		#     print("gradient>>",para.grad)
		# if (batch_idx+1) % opt.accumulation_steps == 0:             # Wait for several backward steps
		# 	optimizer.step()                            # Now we can do an optimizer step
		# 	optimizer.zero_grad()  
		
		if batch_idx % opt.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_dataloader)) + '] ' + \
					 '[Total Loss = ' + '{:.4f}'.format(loss.item()) + '] ' +\
					 '[MSE Loss = ' + '{:.4f}'.format(mse_criterion(torch.stack(recon_combined_seq,dim=1), img).item()) + '] ' +\
					 '[CE Loss = ' + '{:.4f}'.format(ce_criterion(scores,target).item()) + '] ' +\
					 '[Learning rate = ' + '{:.8f}'.format(learning_rate) + '] ' + \
					 '[Accuracy = ' + '{:.2f}'.format(acc) + ']'

					  )
	print("Average training reconstruction loss>>",np.mean(np.array(all_trainmseloss)))
	print("Average training cross entropy loss>>",np.mean(np.array(all_trainceloss)))
	print("Average training accuracy>>",np.mean(np.array(all_trainacc)))
	# np.savez('predictions/{}_eval_tcn_shuffling_9slots_nolrdecay_rowcolposemb_raven_allconfigs_train_images_recons_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy() )
	# np.savez('predictions/{}_tcn_shuffling_augmentation_9slots_nolrdecay_rowcolposemb_raven_allconfigs_train_images_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,masks= masks.cpu().detach().numpy() )

	slot_model.eval()
	transformer_scoring_model.eval()

	
	
	all_valacc = []
   
	# total_train_images = torch.Tensor().to(device).float()
	# total_train_recons_combined = torch.Tensor().to(device).float()
	# total_train_recons = torch.Tensor().to(device).float()
	# total_train_masks = torch.Tensor().to(device).float()
	
	for batch_idx, (val_img,val_target) in enumerate(val_dataloader):
		# print("image and masks and target shape>>",img.shape,masks.shape,target.shape)
# 		# print(torch.max(img), torch.min(img),img.shape)
		
		val_img = val_img.to(device).float() #.unsqueeze(1).float()
		# masks = masks.to(device).float()
		val_target = val_target.to(device)
		val_slots_seq =[]
		# recon_combined_seq =[]
		# recons_seq=[]
		# masks_seq=[]
		for idx in range(val_img.shape[1]):
			# print(idx)
			# print(img[:,idx].shape,masks[:,idx].shape)
		
			# slots = slotmask_model(img[:,idx],masks[:,idx],device)
			# # print("mask max and min>>", torch.max(masks), torch.min(masks))
			# slots_seq.append(slots)
			
			# del slots
			val_recon_combined, val_recons, val_masks, val_slots,val_attn = slot_model(val_img[:,idx],device)
			val_slots_seq.append(val_slots)
			# recon_combined_seq.append(recon_combined)
			# recons_seq.append(recons)
			# masks_seq.append(masks)
			del val_recon_combined,val_recons, val_masks, val_slots, val_attn

		given_panels = torch.stack(val_slots_seq,dim=1)[:,:8]
		answer_panels = torch.stack(val_slots_seq,dim=1)[:,8:]
		# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

		scores = transformer_scoring_model(given_panels,answer_panels,device)
		# print("scores and target>>",scores,target)
		pred = scores.argmax(1)
		acc = torch.eq(pred,val_target).float().mean().item() * 100.0
		all_valacc.append(acc)
	print("Average validation accuracy>>>",np.mean(np.array(all_valacc)))
	# np.savez('predictions/{}_eval_tcn_shuffling_9slots_nolrdecay_rowcolposemb_raven_allconfigs_val_images_recons_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy() )
	# np.savez('predictions/{}_tcn_shuffling_augmentation_9slots_nolrdecay_rowcolposemb_raven_allconfigs_val_images_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,masks= masks.cpu().detach().numpy() )

	if np.mean(np.array(all_valacc)) > max_val_acc:
		print("Validation accuracy increased from %s to %s"%(max_val_acc,np.mean(np.array(all_valacc))))
		max_val_acc = np.mean(np.array(all_valacc))
		print("Saving model$$$$")
		save_nw(slot_model,transformer_scoring_model,optimizer,epoch,'{}_1000weightmse_tcn_shuffling_augmentations_9slots_morewarmup_nolrdecay_0.1dropout_rowcolposemb_raven_allconfigs_run_{}_best.pth.tar'.format(opt.model_name,opt.run),opt.save_path)
		# save_nw(slot_model,transformer_scoring_model,optimizer,epoch,'{}_1000weightmse_rowwisetransformer_colposemb_tcn_secondtransformer_rowposemb_cls_shuffling_augmentations_9slots_morewarmup_nolrdecay_raven_allconfigs_run_{}_best.pth.tar'.format(opt.model_name,opt.run))
		
		# np.savez('predictions/{}_1000weightmse_tcn_shuffling_augmentations_9slots_dspritesdecoder_morewarmup_nolrdecay_0.1dropout_rowcolposemb_raven_allconfigs_train_images_recons_masks_attn_run_{}.npz'.format(opt.model_name,opt.run), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy(),attention= torch.stack(attn_seq,dim=1).cpu().detach().numpy() )
		# np.savez('predictions/{}_1000weightmse_rowwisetransformer_colposemb_tcn_secondtransformer_rowposemb_cls_shuffling_augmentations_9slots_dspritesdecoder_morewarmup_nolrdecay_raven_allconfigs_train_images_recons_masks_attn.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy(),attention= torch.stack(attn_seq,dim=1).cpu().detach().numpy() )

		all_testacc = []
   
	# total_train_images = torch.Tensor().to(device).float()
	# total_train_recons_combined = torch.Tensor().to(device).float()
	# total_train_recons = torch.Tensor().to(device).float()
	# total_train_masks = torch.Tensor().to(device).float()
	
		for batch_idx, (img,target) in enumerate(test_dataloader):
			# print("image and masks and target shape>>",img.shape,masks.shape,target.shape)
	# 		# print(torch.max(img), torch.min(img),img.shape)
			
			img = img.to(device).float() #.unsqueeze(1).float()
			# masks = masks.to(device).float()
			target = target.to(device)
			slots_seq =[]
			# recon_combined_seq =[]
			# recons_seq=[]
			# masks_seq=[]
			
			for idx in range(img.shape[1]):
				# print(idx)
				# print(img[:,idx].shape,masks[:,idx].shape)
			
				# slots = slotmask_model(img[:,idx],masks[:,idx],device)
				# # print("mask max and min>>", torch.max(masks), torch.min(masks))
				# slots_seq.append(slots)
				
				# del slots
				recon_combined, recons, masks, slots,attn = slot_model(img[:,idx],device)
				slots_seq.append(slots)
				# recon_combined_seq.append(recon_combined)
				# recons_seq.append(recons)
				# masks_seq.append(masks)
				del recon_combined,recons, masks, slots,attn

			given_panels = torch.stack(slots_seq,dim=1)[:,:8]
			answer_panels = torch.stack(slots_seq,dim=1)[:,8:]
			# print("given and answer panels shape>>>", given_panels.shape, answer_panels.shape)

			scores = transformer_scoring_model(given_panels,answer_panels,device)
			# print("scores and target>>",scores,target)
			pred = scores.argmax(1)
			acc = torch.eq(pred,target).float().mean().item() * 100.0
			all_testacc.append(acc)

		print("Average test accuracy>>>",np.mean(np.array(all_testacc)))
		# np.savez('predictions/{}_eval_tcn_shuffling_9slots_nolrdecay_rowcolposemb_raven_allconfigs_test_images_recons_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy() )

	else:
		print("Validation accuracy didn't increase and hence skipping model saving!!!!")
	print("Best validation accuracy till now >>",max_val_acc)
	

# 	print("Average training reconstruction loss>>",np.mean(np.array(all_trainloss)))
# # 	np.savez('predictions/{}_15slots_pgm_neutral_train_images_recons_masks.npz'.format(opt.model_name), images= total_train_images.cpu().detach().numpy() ,recon_combined = total_train_recons_combined.cpu().detach().numpy(),recons = total_train_recons.cpu().detach().numpy(),masks= total_train_masks.cpu().detach().numpy() )
		# if batch_idx % opt.save_interval ==0:
		# 	np.savez('predictions/{}_multigpu_eval_16slots_moreweightmse_dspritesdecoder_nolrdecay_rowcolposemb_pgm_neutral_test_images_recons_masks.npz'.format(opt.model_name), images= img.cpu().detach().numpy() ,recon_combined = torch.stack(recon_combined_seq,dim=1).cpu().detach().numpy(),recons = torch.stack(recons_seq,dim=1).cpu().detach().numpy(),masks= torch.stack(masks_seq,dim=1).cpu().detach().numpy() )

			# save_nw(slot_model,transformer_scoring_model,optimizer,epoch,'{}_16slots_dspritesdecoder_nolrdecay_rowcolposemb_pgm_wholetrainset_extrapolation_run_{}_best.pth.tar'.format(opt.model_name,opt.run))

		
