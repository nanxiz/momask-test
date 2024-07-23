import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F
from openai import OpenAI

#from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
#from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed

from utils.motion_process import recover_from_ric
#from utils.plot_script import plot_3d_motion


import numpy as np
from models.vq.model import RVQVAE, LengthEstimator
from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer

from scipy.spatial.transform import Rotation as R
from typing import Union
import runpod
import logging
import subprocess
import time
import requests

clip_version = 'ViT-B/32'

MAX_SERVER_START_WAIT_S = 200

logging.basicConfig(level=logging.INFO)

model_folder_path = "Phi-3-mini-128k-instruct-awq"

def start_vllm_server():
    if not os.path.exists(model_folder_path):
        raise FileNotFoundError(f"Model directory not found at {model_folder_path}")
    
    process = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_folder_path,
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "4095",
        "--quantization", "awq",
        "--trust-remote-code"
    ])

def wait_for_server():
    start = time.time()
    while True:
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                logging.info("Server is ready.")
                break
        except requests.exceptions.RequestException as err:
            if time.time() - start > MAX_SERVER_START_WAIT_S:
                logging.error("Server failed to start in time.", exc_info=True)
                raise RuntimeError("Server failed to start in time.") from err
            logging.info("Server is not ready yet. Retrying...")
            time.sleep(0.5)
    

start_vllm_server()
wait_for_server()

openai_api_key = "hihi"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


def load_trans_model(model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer



joint_names = [
    "mixamorig:Hips", "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg", "mixamorig:Spine",
    "mixamorig:LeftLeg", "mixamorig:RightLeg", "mixamorig:Spine1", "mixamorig:LeftFoot",
    "mixamorig:RightFoot", "mixamorig:Spine2", "mixamorig:LeftToeBase", "mixamorig:RightToeBase",
    "mixamorig:Neck", "mixamorig:LeftShoulder", "mixamorig:RightShoulder", "mixamorig:Head",
    "mixamorig:LeftArm", "mixamorig:RightArm", "mixamorig:LeftForeArm", "mixamorig:RightForeArm",
    "mixamorig:LeftHand", "mixamorig:RightHand"
]

# 初始化额外旋转的字典
def initialize_rotations():
    additional_rotations = {}
    additional_rotations["mixamorig:Head"] = R.from_euler('xyz', [-12, 0, 0], degrees=True)
    additional_rotations["mixamorig:LeftShoulder"] = R.from_euler('xyz', [12.2, -2.2, -1], degrees=True)
    additional_rotations["mixamorig:RightShoulder"] = R.from_euler('xyz', [10, -63, -15], degrees=True)
    additional_rotations["mixamorig:LeftArm"] = R.from_euler('xyz', [-2.9, -0.517, -0.534], degrees=True)
    additional_rotations["mixamorig:RightArm"] = R.from_euler('xyz', [-4.055, -5.536, -4.413], degrees=True)
    additional_rotations["mixamorig:LeftFoot"] = R.from_euler('xyz', [9, 0, 5], degrees=True)
    additional_rotations["mixamorig:RightFoot"] = R.from_euler('xyz', [7, 0, -3], degrees=True)
    additional_rotations["mixamorig:RightUpLeg"] = R.from_euler('xyz', [0, -6, -7], degrees=True)
    additional_rotations["mixamorig:LeftUpLeg"] = R.from_euler('xyz', [0, 6, 7], degrees=True)
    additional_rotations["mixamorig:LeftLeg"] = R.from_euler('xyz', [0, -2, -3], degrees=True)
    additional_rotations["mixamorig:RightLeg"] = R.from_euler('xyz', [0, 2, 3], degrees=True)
    return additional_rotations

# 应用额外的旋转
def apply_additional_rotation(joint_name, base_rotation, additional_rotations):
    if joint_name in additional_rotations:
        # 四元数乘法
        #return additional_rotations[joint_name] * base_rotation
        return  base_rotation * additional_rotations[joint_name] 
    return base_rotation

# 对所有帧应用额外旋转
def apply_additional_rotation_to_all_frames_quats(quats, joint_names, additional_rotations):
    num_frames = quats.shape[0]
    num_joints = quats.shape[1]
    rotated_quats = np.empty_like(quats)

    for frame in range(num_frames):
        for joint in range(num_joints):
            joint_name = joint_names[joint]
            base_rotation = R.from_quat(quats[frame, joint])
            rotated_quats[frame, joint] = apply_additional_rotation(joint_name, base_rotation, additional_rotations).as_quat()
    return rotated_quats

additional_rotations = initialize_rotations()

INIT_POSITIONS = np.array([[[ 0.000000e+00,  1.850000e+00,  5.960000e-07],
       [-1.632465e-01,  1.762824e+00,  1.966429e-02],
       [ 1.632465e-01,  1.762824e+00,  2.104044e-02],
       [ 1.000000e-09,  2.006599e+00, -9.250164e-03],
       [-1.484674e-01,  9.308408e-01,  1.281303e-02],
       [ 1.484749e-01,  9.312698e-01, -8.877754e-03],
       [-1.000000e-09,  2.189296e+00, -2.004290e-02],
       [-1.339307e-01,  1.124900e-01, -3.766675e-02],
       [ 1.293448e-01,  1.143031e-01, -8.162399e-02],
       [-2.000000e-09,  2.398094e+00, -3.237748e-02],
       [-1.928200e-01, -9.409383e-03,  1.737747e-01],
       [ 1.922907e-01, -2.384451e-02,  1.430029e-01],
       [-4.000000e-09,  2.632992e+00, -4.625392e-02],
       [-6.867484e-02,  2.602175e+00, -4.561949e-02],
       [ 6.867483e-02,  2.602175e+00, -4.688835e-02],
       [-2.000000e-09,  2.750809e+00, -3.047061e-02],
       [-1.996271e-01,  2.526263e+00, -4.466033e-02],
       [ 1.992011e-01,  2.525531e+00, -4.617119e-02],
       [-3.265694e-01,  2.094378e+00, -7.912588e-02],
       [ 3.138946e-01,  2.088543e+00, -7.288742e-02],
       [-4.583911e-01,  1.645335e+00, -5.935407e-02],
       [ 4.318638e-01,  1.635855e+00, -4.882717e-02]]])



INIT_GLOBAL_ROTATIONS = np.array([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
       [ 1.8000e-04, -4.1200e-03,  9.9995e-01, -8.8800e-03],
       [-1.2000e-04, -1.7980e-02,  9.9980e-01,  8.8800e-03],
       [-2.9500e-02,  0.0000e+00,  0.0000e+00,  9.9956e-01],
       [-5.2450e-02, -3.1220e-02,  9.9811e-01, -7.2400e-03],
       [ 5.2700e-02, -4.4940e-02,  9.9756e-01,  9.3100e-03],
       [-2.9500e-02,  0.0000e+00,  0.0000e+00,  9.9956e-01],
       [-1.0584e-01,  5.0160e-01,  8.5531e-01,  7.5050e-02],
       [ 1.0641e-01,  4.8961e-01,  8.6224e-01, -7.4210e-02],
       [-2.9500e-02,  0.0000e+00,  0.0000e+00,  9.9956e-01],
       [-5.4810e-02,  6.9976e-01,  7.0452e-01,  1.0482e-01],
       [ 5.8150e-02,  6.7531e-01,  7.2806e-01, -1.0253e-01],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
       [ 5.9681e-01, -3.4121e-01,  6.2815e-01,  3.6445e-01],
       [ 5.9609e-01,  3.3925e-01, -6.3075e-01,  3.6295e-01],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
       [ 6.9653e-01, -1.2749e-01,  7.0225e-01,  7.3740e-02],
       [ 7.0386e-01,  1.1172e-01, -6.9811e-01,  6.8870e-02],
       [ 7.0287e-01, -8.5980e-02,  6.9664e-01,  1.1524e-01],
       [ 7.0895e-01,  7.2690e-02, -6.9325e-01,  1.0730e-01],
       [ 6.0636e-01, -1.0161e-01,  7.8287e-01,  9.5510e-02],
       [ 6.2603e-01,  8.2400e-02, -7.7063e-01,  8.6220e-02]])

parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
naive_hybrik = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
child = [-1, 4, 5, 6, 7, 8, 9, 10, 11, -1, -2, -2, 15, 16, 17, -2, 18, 19, 20, 21, -2, -2]
num_nodes = 22

'''
# Download model if needed
if not os.path.exists("/data/checkpoints/t2m"):
    os.system("bash prepare/download_models_demo.sh")
if not os.path.exists("checkpoints/t2m"):
    os.system("ln -s /data/checkpoints checkpoints")
if not os.path.exists("/data/stats"):
    os.makedirs("/data/stats")
    with open("/data/stats/Prompts.text", 'w') as f:
        pass
'''

parser = EvalT2MOptions()
opt = parser.parse()
fixseed(opt.seed)

# Path to source motion
SOURCE_MOTION = "doublefinalmotionref263.npy"
opt.source_motion = SOURCE_MOTION

opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
torch.autograd.set_detect_anomaly(True)

dim_pose = 251 if opt.dataset_name == 'kit' else 263

root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
model_dir = pjoin(root_dir, 'model')

result_dir = pjoin('./editing', opt.ext)
joints_dir = pjoin(result_dir, 'joints')
animation_dir = pjoin(result_dir, 'animations')
os.makedirs(joints_dir, exist_ok=True)
os.makedirs(animation_dir,exist_ok=True)

model_opt_path = pjoin(root_dir, 'opt.txt')
model_opt = get_opt(model_opt_path, device=opt.device)


#######################
######Loading RVQ######
#######################
vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
vq_opt = get_opt(vq_opt_path, device=opt.device)
vq_opt.dim_pose = dim_pose
vq_model, vq_opt = load_vq_model(vq_opt)

model_opt.num_tokens = vq_opt.nb_code
model_opt.num_quantizers = vq_opt.num_quantizers
model_opt.code_dim = vq_opt.code_dim

#################################
######Loading R-Transformer######
#################################
res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
res_opt = get_opt(res_opt_path, device=opt.device)
res_model = load_res_model(res_opt, vq_opt, opt)

assert res_opt.vq_name == model_opt.vq_name

#################################
######Loading M-Transformer######
#################################
t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

t2m_transformer.eval()
vq_model.eval()
res_model.eval()

res_model.to(opt.device)
t2m_transformer.to(opt.device)
vq_model.to(opt.device)

def inv_transform(data):
    return data * std + mean

#################################
######Set max motion length######
#################################
max_motion_length = 196
mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

motion = np.load(opt.source_motion)
m_length = len(motion)
motion = (motion - mean) / std
if max_motion_length > m_length:
    motion = np.concatenate([motion, np.zeros((max_motion_length - m_length, motion.shape[1])) ], axis=0)
motion = torch.from_numpy(motion)[None].to(opt.device)


def joints_pos_inference(text_prompt, mask_edit_section, source_motion = opt.source_motion, use_res_model = True):
    motion = np.load(source_motion)     
    m_length = len(motion)
    motion = (motion - mean) / std
    if max_motion_length > m_length:
        motion = np.concatenate([motion, np.zeros((max_motion_length - m_length, motion.shape[1])) ], axis=0)
    motion = torch.from_numpy(motion)[None].to(opt.device)


    prompt_list = []
    length_list = []
    if opt.motion_length == 0:
        opt.motion_length = m_length
        print("Using default motion length.")
    
    prompt_list.append(text_prompt)
    length_list.append(opt.motion_length)
    if text_prompt == "":
        raise "Using an empty text prompt."

    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(opt.device).long()


    m_length = token_lens * 4
    captions = prompt_list
    print_captions = captions[0]

    _edit_slice = mask_edit_section
    print(_edit_slice)
    edit_slice = []
    for eds in _edit_slice:
        _start, _end = eds.split(',')
        _start = eval(_start)
        _end = eval(_end)
        edit_slice.append([_start, _end])

    #sample = 0

    with torch.no_grad():
        tokens, features = vq_model.encode(motion)
    ### build editing mask, TOEDIT marked as 1 ###
    edit_mask = torch.zeros_like(tokens[..., 0])
    seq_len = tokens.shape[1]
    for _start, _end in edit_slice:
        if isinstance(_start, float):
            _start = int(_start*seq_len)
            _end = int(_end*seq_len)
        else:
            _start //= 4
            _end //= 4
        edit_mask[:, _start: _end] = 1
        print_captions = f'{print_captions} [{_start*4/20.}s - {_end*4/20.}s]'
    edit_mask = edit_mask.bool()

    
    with torch.no_grad():
        mids = t2m_transformer.edit(
                                        captions, tokens[..., 0].clone(), m_length//4,
                                        timesteps=opt.time_steps,
                                        cond_scale=opt.cond_scale,
                                        temperature=opt.temperature,
                                        topk_filter_thres=opt.topkr,
                                        gsample=opt.gumbel_sample,
                                        force_mask=opt.force_mask,
                                        edit_mask=edit_mask.clone(),
                                        )
        if use_res_model:
            mids = res_model.generate(mids, captions, m_length//4, temperature=1, cond_scale=2)
        else:
            mids.unsqueeze_(-1)

        pred_motions = vq_model.forward_decoder(mids)

        pred_motions = pred_motions.detach().cpu().numpy()

        source_motions = motion.detach().cpu().numpy()

        data = inv_transform(pred_motions)
        source_data = inv_transform(source_motions)

    for k, (caption, joint_data, source_data)  in enumerate(zip(captions, data, source_data)):
        print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
        animation_path = pjoin(animation_dir, str(k))
        joint_path = pjoin(joints_dir, str(k))

        os.makedirs(animation_path, exist_ok=True)
        os.makedirs(joint_path, exist_ok=True)

        joint_data = joint_data[:m_length[k]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

        actual_source = np.load("doublefinalmotionrefjoints.npy")
        soucre_joint = actual_source

        difference = soucre_joint[0] - joint[0]
        joint = joint + difference

        #source_data = source_data[:m_length[k]]
        #soucre_joint = recover_from_ric(torch.from_numpy(source_data).float(), 22).numpy()

        #bvh_path = pjoin(animation_path, "sample%d_len%d_ik.bvh"%(k, m_length[k]))
        #_, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

        #bvh_path = pjoin(animation_path, "sample%d_len%d.bvh" % (k, m_length[k]))
        #_, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)


        #save_path = pjoin(animation_path, "sample%d_len%d.mp4"%(k, m_length[k]))
        #ik_save_path = pjoin(animation_path, "sample%d_len%d_ik.mp4"%(k, m_length[k]))
        #source_save_path = pjoin(animation_path, "sample%d_source_len%d.mp4"%(k, m_length[k]))

        #plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=print_captions, fps=20)
        #plot_3d_motion(save_path, kinematic_chain, joint, title=print_captions, fps=20)
        #plot_3d_motion(source_save_path, kinematic_chain, soucre_joint, title='None', fps=20)
        #np.save(pjoin(joint_path, "sample%d_len%d.npy"%(k, m_length[k])), joint)
        #np.save(pjoin(joint_path, "sample%d_len%d_ik.npy"%(k, m_length[k])), ik_joint)

        #ik_joints_json = json.dumps(ik_joint.tolist())
    return joint

def pair_intervals(input_string):
    # Split the string by commas
    elements = input_string.split(',')
    
    # Check if the number of elements is even
    if len(elements) % 2 != 0:
        return ["10,100"]
    
    # Group elements into pairs
    paired_intervals = [",".join(elements[i:i+2]) for i in range(0, len(elements), 2)]
    
    return paired_intervals

#CONVERT JOINTS TO QUATS

def multi_child_rot(t, p, pose_global_parent):

    m = np.matmul(t, np.transpose(np.matmul(np.linalg.inv(pose_global_parent), p), [0, 2, 1]))
    u, s, vt = np.linalg.svd(m)
    r = np.matmul(np.transpose(vt, [0, 2, 1]), np.transpose(u, [0, 2, 1]))
    err_det_mask = (np.linalg.det(r) < 0.0).reshape(-1, 1, 1)
    id_fix = np.reshape(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                            [1, 3, 3])
    r_fix = np.matmul(np.transpose(vt, [0, 2, 1]),
                          np.matmul(id_fix,
                                    np.transpose(u, [0, 2, 1])))
    r = r * (1.0 - err_det_mask) + r_fix * err_det_mask
    return r, np.matmul(pose_global_parent, r)

def single_child_rot(t, p, pose_global_parent, twist=None):
    p_rot = np.matmul(np.linalg.inv(pose_global_parent), p)
        
        
    cross = np.cross(t, p_rot, axisa=1, axisb=1, axisc=1)
    sina = np.linalg.norm(cross, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                               np.linalg.norm(p_rot, axis=1, keepdims=True))
    cross = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    cosa = np.sum(t * p_rot, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                           np.linalg.norm(p_rot, axis=1, keepdims=True))
    sina = np.reshape(sina, [-1, 1, 1])
    cosa = np.reshape(cosa, [-1, 1, 1])
    skew_sym_t = np.stack([0.0 * cross[:, 0], -cross[:, 2], cross[:, 1],
                               cross[:, 2], 0.0 * cross[:, 0], -cross[:, 0],
                               -cross[:, 1], cross[:, 0], 0.0 * cross[:, 0]], 1)
    skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
    dsw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                 skew_sym_t)
    if twist is not None:
        skew_sym_t = np.stack([0.0 * t[:, 0], -t[:, 2], t[:, 1],
                                   t[:, 2], 0.0 * t[:, 0], -t[:, 0],
                                   -t[:, 1], t[:, 0], 0.0 * t[:, 0]], 1)
        skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
        sina = np.reshape(twist[:, 1], [-1, 1, 1])
        cosa = np.reshape(twist[:, 0], [-1, 1, 1])
        dtw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                    ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                     skew_sym_t)
        dsw_rotmat = np.matmul(dsw_rotmat, dtw_rotmat)
    return dsw_rotmat, np.matmul(pose_global_parent, dsw_rotmat)
    
def convert_rotmats_to_quats(rotmats):
    """
    Converts an array of rotation matrices to an array of quaternions.

    Parameters:
    rotmats (numpy.ndarray): An array of shape (n, 22, 3, 3) containing rotation matrices.

    Returns:
    numpy.ndarray: An array of shape (n, 22, 4) containing quaternions.
    """
    n, num_joints, _, _ = rotmats.shape
    quats = np.empty((n, num_joints, 4))

    for i in range(n):
        for j in range(num_joints):
            rotation = R.from_matrix(rotmats[i, j])
            quats[i, j] = rotation.as_quat()

    return quats

# Quaternion to rotation matrix and Euler angles conversion
def quat_to_rot_matrix_euler(quaternion):
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion)
    rot_matrix = rotation.as_matrix()

    # Convert rotation matrix to Euler angles (in degrees)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    
    return rot_matrix, euler_angles

def inference(event) -> Union[str, dict]:
    input_data = event.get("input", {})
    # text_prompt = input_data.get("text_prompt", "")
    mask_edit_section = input_data.get("mask_edit_section", "")
    init_positions = input_data.get("init_positions", INIT_POSITIONS)
    init_global_rotations = input_data.get("init_global_rotations", INIT_GLOBAL_ROTATIONS)
    user_text = input_data.get("user", "")
    bot_text = input_data.get("bot", "")

    if user_text == "":
        logging.info("Received an empty 'user' input.")
    if bot_text == "":
        logging.info("Received an empty 'bot' input.")

    speaker_text = f"Speaker 1: {user_text} Speaker 2: {bot_text}"
    messages = [
        {"role": "system", "content": "You will be given a conversation between two speakers. Both speakers are standing. You need to imagine the body motion Speaker 2 (a cute boy) has when he is speaking. You need to describe the body motiion in one sentence using concise language. Note that only body and head movement of Speaker 2 should be included, and not respond with anything related to facial expression. You should just describe the pure movement clearly, don't add anything uncertain. The description should start with 'A person'. Here's an example you could generate: 'A person stands for a few seconds and picks up its arms and shakes them.'. Now, be creative, and here's their conversation:"},
        {"role": "system", "content":speaker_text}

    ]


    extra_body = {
        "temperature": 0.53,
        "max_tokens": 128,
    }

    response = client.chat.completions.create(
        model=model_folder_path, 
        messages=messages,
        extra_body=extra_body
    )
    text_prompt = response.choices[0].message.content if response.choices else "A person is standing."

    if not text_prompt:
        return {"error": "text_prompt is required."}

    if not mask_edit_section:
        return {"error": "mask_edit_section is required."}
    
    mask_edit_section = pair_intervals(mask_edit_section)
    
    motion = joints_pos_inference(text_prompt, mask_edit_section)
    initial_global_rot_matrix , initial_global_rot_angle = quat_to_rot_matrix_euler(init_global_rotations)
    
    positions = motion.copy()  # Example data, replace with your actual data
    positions[:, :, 0] *= -1
    #parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    relative_joint_pos = init_positions - init_positions[:, parents]
    relative_joint_pos[0][0] = init_positions[0][0].copy()
    flattened_relative_joint_pos = relative_joint_pos.flatten()
    #naive_hybrik = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    #num_nodes = 22
    #child = [-1, 4, 5, 6, 7, 8, 9, 10, 11, -1, -2, -2, 15, 16, 17, -2, 18, 19, 20, 21, -2, -2]
    #verified
    bones = np.reshape(np.array(flattened_relative_joint_pos), [22, 3])[:num_nodes]

    batch_size = np.shape(positions)[0]
    joints_rel = positions - positions[:, parents]
    joints_hybrik = 0.0 * joints_rel
    pose_global = np.zeros([batch_size, num_nodes, 3, 3])
    pose = np.zeros([batch_size, num_nodes, 3, 3])
    n_frame = positions.shape[0]
    init_root_loc = np.array(init_positions[0][0].copy())
    default_root_loc = np.tile(init_root_loc, (n_frame, 1))

    for i in range(num_nodes):
        if i == 0:
            joints_hybrik[:, 0] = default_root_loc
        else:
            joints_hybrik[:, i] = np.matmul(pose_global[:, parents[i]],
                                        np.reshape(bones[i], [1, 3, 1])).reshape(-1, 3) + joints_hybrik[:, parents[i]]
        if child[i] == -2:
            pose[:, i] = pose[:, i] + np.eye(3).reshape(1, 3, 3)
            pose_global[:, i] = pose_global[:, parents[i]]
            continue
        if i == 0:
            r, rg = multi_child_rot(np.transpose(bones[[1, 2, 3]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [1, 2, 3]], [0, 2, 1]),
                                             np.eye(3).reshape(1, 3, 3))

        elif i == 9:
            r, rg = multi_child_rot(np.transpose(bones[[12, 13, 14]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [12, 13, 14]], [0, 2, 1]),
                                             pose_global[:, parents[9]])
        else:
                
            p = joints_rel[:, child[i]]
            if naive_hybrik[i] == 0:
                p = positions[:, child[i]] - positions[:, i]
            twi = None
            r, rg = single_child_rot(bones[child[i]].reshape(1, 3, 1),
                                              p.reshape(-1, 3, 1),
                                              pose_global[:, parents[i]],
                                              twi)
        pose[:, i] = r
        pose_global[:, i] = rg
        
    res_global_rot = pose_global.copy()
    # apply rotation difference to initial global rotation to get final global rotations
    # alternative, do this in Unity
    for i in range(pose_global.shape[1]): 
        res_global_rot[:, i] = np.matmul(res_global_rot[:, i], initial_global_rot_matrix[i])

    res_global_rot_quats = convert_rotmats_to_quats(res_global_rot)
    altered_res_global_rot_quats = apply_additional_rotation_to_all_frames_quats(res_global_rot_quats, joint_names, additional_rotations)
    
    hip_pos = joints_rel[:, 0, :]

    return {"text_prompt": text_prompt, "global_quats": altered_res_global_rot_quats.tolist(), "hip_pos": hip_pos.tolist(), "length": n_frame}

runpod.serverless.start({"handler": inference})
    
'''
if __name__ == "__main__":
    user = "Hey you!"
    bot = "whats up my friend? why are you here?"
    mask_edit_section = "0.2,0.8"  # 开始时间和结束时间
    #source_motion = "path_to_source_motion.npy"  # 这里应该指向一个有效的.npy文件
    #use_res_model = True

    # 调用 inference 函数
    #result = joints_pos_inference(text_prompt, mask_edit_section)
    #print(result)

    event = {
        "input": {
            "user": user,
            "bot": bot,
            "mask_edit_section": mask_edit_section,
            #"init_positions": init_positions,
            #"init_global_positions": init_global_rotations
        }
    }
    result = inference(event)
    joint = np.array(result.get("global_quats",[]))
    
    print(joint.shape)

'''
