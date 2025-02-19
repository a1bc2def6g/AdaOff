import torch
import time
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


h = 4096
# Define the size of the matrices
matrix_size_1 = (1, h)
matrix_size_2 = (h, h)

# Create random matrices
cpu_matrix1 = torch.randn(matrix_size_1)
cpu_matrix2 = torch.randn(matrix_size_2)

class CustomLinear(nn.Module):
    def __init__(self, embed_dim=4096, ffn_dim=11008, weight_init=None, bias_init=None):
        super(CustomLinear, self).__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        
        # Initialize weights and biases
        self.w_q = nn.Parameter(torch.Tensor(embed_dim, embed_dim).half())
        self.w_k = nn.Parameter(torch.Tensor(embed_dim, embed_dim).half())
        self.w_v = nn.Parameter(torch.Tensor(embed_dim, embed_dim).half())
        self.w_o = nn.Parameter(torch.Tensor(embed_dim, embed_dim).half())
        self.attn_norm = nn.Parameter(torch.Tensor(embed_dim).half())
        
        
        nn.init.kaiming_uniform_(self.w_q, a=math.sqrt(5)) 
        nn.init.kaiming_uniform_(self.w_k, a=math.sqrt(5))  
        nn.init.kaiming_uniform_(self.w_v, a=math.sqrt(5))  
        nn.init.kaiming_uniform_(self.w_o, a=math.sqrt(5))  
        nn.init.uniform_(self.attn_norm, -1, 1)  
        
        self.w_1 = nn.Parameter(torch.Tensor(ffn_dim, embed_dim).half())
        self.w_2 = nn.Parameter(torch.Tensor(embed_dim, ffn_dim).half())
        self.mlp_norm = nn.Parameter(torch.Tensor(embed_dim).half())
        nn.init.kaiming_uniform_(self.w_1, a=math.sqrt(5))  
        nn.init.kaiming_uniform_(self.w_2, a=math.sqrt(5))  
        nn.init.uniform_(self.mlp_norm, -1, 1)  
        
    def forward(self, input, K, V):
        hidden = F.layer_norm(input, (self.embed_dim,), weight=self.attn_norm)
        q = F.linear(hidden, self.w_q)
        k = F.linear(hidden, self.w_k)
        v = F.linear(hidden, self.w_v)

        # q_i = q.view(1,1,32,128)
        # v_i = v.view(1,1,32,128)
        # k_i = k.view(1,1,32,128)

        K = torch.concat((K,k),dim=0)
        V = torch.concat((V,v),dim=0)

        attn_w = torch.mm(q,K.T)
        attn_weights = F.softmax(attn_w, dim=1)
        value = torch.mm(attn_weights,V)
        value = F.linear(value, self.w_o)
        value.add_(input)

        hidden = F.layer_norm(value, (self.embed_dim,), weight=self.mlp_norm)
        out = F.linear(hidden, self.w_1)
        F.relu(out, inplace=True)
        out = F.linear(out, self.w_2)
        out.add_(hidden)

        return out

# Compute matrix multiplication time on CPU
def run_cpu(matrix1, matrix2, num_runs=10):
    start_time = time.time()
    for _ in range(num_runs):
        result = torch.matmul(matrix1, matrix2)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

# Compute matrix multiplication time on GPU (separating transfer time and computation time)
def run_gpu(matrix1, matrix2, num_runs=10):
    transfer_times = []
    transfer_times_2 = []
    compute_times = []
    
    for _ in range(num_runs):
        # Record transfer start time
        transfer_start_time = time.time()
        
        # Transfer matrices to GPU
        gpu_matrix1 = matrix1.to('cuda')
        torch.cuda.synchronize()  # synchronize
        transfer_end_time = time.time()
        transfer_times.append(transfer_end_time - transfer_start_time)
        transfer_end_time = time.time()

        gpu_matrix2 = matrix2.to('cuda')
        
        # Record transfer end time
        torch.cuda.synchronize()  # synchronize
        transfer_end_time_2 = time.time()
        transfer_times_2.append(transfer_end_time_2 - transfer_end_time)
        
        # Record computation start time
        compute_start_time = time.time()
        
        # Perform matrix multiplication
        result = torch.matmul(gpu_matrix1, gpu_matrix2)
        
        # Record computation end time
        torch.cuda.synchronize()  
        compute_end_time = time.time()
        compute_times.append(compute_end_time - compute_start_time)
    
    avg_transfer_time_1 = sum(transfer_times) / num_runs
    avg_transfer_time_2 = sum(transfer_times_2) / num_runs
    avg_compute_time = sum(compute_times) / num_runs
    return avg_transfer_time_1, avg_transfer_time_2, avg_compute_time

def run_cpu_nn(module, input, K, V, num_runs=10):
    start_time = time.time()
    for _ in range(num_runs):
        result = module(input,K,V)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def run_gpu_nn(module, input, K, V, num_runs=10):
    transfer_times = []
    transfer_times_2 = []
    compute_times = []
    
    for _ in range(num_runs):
        transfer_start_time = time.time()
        
        module = module.cuda()
        torch.cuda.synchronize() 
        transfer_end_time = time.time()
        transfer_times.append(transfer_end_time - transfer_start_time)
        transfer_end_time = time.time()

        input_ = input.to('cuda')
        
        
        torch.cuda.synchronize()  
        transfer_end_time_2 = time.time()
        transfer_times_2.append(transfer_end_time_2 - transfer_end_time)
        K_ = K.to('cuda')
        V_ = V.to('cuda')
        
        compute_start_time = time.time()
        
        result = module(input_,K_,V_)
        
        torch.cuda.synchronize() 
        compute_end_time = time.time()
        compute_times.append(compute_end_time - transfer_start_time)
        module = module.cpu()
    
    avg_transfer_time_1 = sum(transfer_times) / num_runs
    avg_transfer_time_2 = sum(transfer_times_2) / num_runs
    avg_compute_time = sum(compute_times) / num_runs
    return avg_transfer_time_1, avg_transfer_time_2, avg_compute_time




def measure_ddr_to_cpu_bandwidth(buffer_size=1_000_000_000, num_trials=10):
    buffer = np.random.rand(buffer_size // 8).astype(np.float64)  

    total_time = 0.0
    for _ in range(num_trials):
        start_time = time.time()
        
        # Simulate reading data from memory
        _ = buffer.sum()  
        
        end_time = time.time()
        total_time += end_time - start_time

    avg_time = total_time / num_trials
    bandwidth = (buffer_size / avg_time) / (1024 ** 3)  # transfer to GB/s
    return bandwidth


def build_module_and_input(embed_dim=4096,ffn_dim=16384,len=10):
    K = torch.randn((len,embed_dim))
    V = torch.randn((len,embed_dim))
    custom_trans = CustomLinear(embed_dim=embed_dim,ffn_dim=ffn_dim)
    input = torch.randn((1,embed_dim))
    return custom_trans, K, V, input
from collections import OrderedDict

class my_seq(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self,input,K,V):
        for module in self:
            input = module(input,K,V)
        return input
        

def build_big_module_and_input(embed_dim=4096,ffn_dim=16384, len=10, layer=16):
    K = torch.randn((len,embed_dim)).half()
    V = torch.randn((len,embed_dim)).half()
    custom_trans = CustomLinear(embed_dim=embed_dim,ffn_dim=ffn_dim)
    model = my_seq(*[CustomLinear(embed_dim=embed_dim,ffn_dim=ffn_dim) for i in range(layer)])
    input = torch.randn((1,embed_dim)).half()
    return custom_trans, K, V, input

def get_c(embed_dim=4096,layer=16):
    return ((embed_dim*embed_dim*12 + 2*embed_dim)*layer).item()

def find_intersection(m1, b1, m2, b2):
    if m1 == m2:
        return None  

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return x, y


def find_intersection_2d(a1, b1, c1, a2, b2, c2):
    """
    Solve for the intersection of two quadratic curves.
    
    parameters:
    y = a1*x^2 + b1*x + c1
    y = a2*x^2 + b2*x + c2
    
    """
    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    D = b**2 - 4*a*c

    intersections = []

    if D != 0:
        x1 = (-b + np.sqrt(abs(D))) / (2 * a)
        x2 = (-b - np.sqrt(abs(D))) / (2 * a)
        y1 = a1*x1**2 + b1*x1 + c1
        y2 = a1*x2**2 + b1*x2 + c1
        intersections.append((x1, y1))
        intersections.append((x2, y2))
    elif D == 0:
        x = -b / (2 * a)
        y = a1*x**2 + b1*x + c1
        intersections.append((x, y))

    return intersections

def find_intersection_1_2d(m, b, a, b_quad, c):
    # Find the intersection of a line and a quadratic curve
    # mx + b = ax^2 + bx + c
    # ax^2 + (b_quad - m)x + (c - b) = 0
    coefficients = [a, b_quad - m, c - b]

    roots = np.roots(coefficients)

    # cal intersection
    intersections = []
    for x in roots:
        if np.isreal(x):  
            y = m * x + b  
            intersections.append((x.real, y))

    return intersections

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# run test on CPU and GPU
rand = 1100
# for rand in np.arange(0,5000,100):
setup_seed(rand)
# analyze(1)
# analyze_nn(40)
thread_num = 1
avg_paras = 0
epochs = 1
hardware_information = {
    'cpu_ap':125,  #GFLOPS
    'gpu_ap':35600,  #GFLOPS
    'dram_bw':25.6, #GB/s
    'pcie_bw':32,
    'gpu_mem_bw':960,
}
for i in range(epochs):
    x_cpu = np.arange(1024,10240,1000,dtype=int)
    x_c_cpu = []
    y_cpu = []
    X_cpu = []
    K_list = []
    V_list = []
    input_list = []
    layer = 8
    for embed_dim in x_cpu:
        kv_params = 10*embed_dim*2
        # embed_dim=512
        c = get_c(embed_dim=embed_dim,layer=layer) #get the total parameters of the build model
        x_c_cpu.append(c)
        cpu_p1 = {'c':c,'h':embed_dim,'t':0}
        start = time.time()
        module,K,V,input = build_big_module_and_input(embed_dim=embed_dim,ffn_dim=embed_dim*4,layer=layer)
        K = torch.randn((10,embed_dim)).half()
        V = torch.randn((10,embed_dim)).half()
        input = torch.randn((1,embed_dim)).half()
        K_list.append(K)
        V_list.append(V)
        input_list.append(input)
        print("cpu_build_moudle_time:",time.time()-start)
        start = time.time()
        cpu_t1 = run_cpu_nn(module, input, K, V, 2)
        print("cpu_run_time:",time.time()-start)
        cpu_p1['t'] = cpu_t1
        X_cpu.append([2*c/hardware_information['cpu_ap'],2*c/hardware_information['dram_bw'],2*embed_dim/hardware_information['pcie_bw'],1])
        y_cpu.append(cpu_t1/thread_num)

    const_cpu = np.linalg.solve(X_cpu[-5:-1],y_cpu[-5:-1])
    

    
    z1 = np.polyfit(x_c_cpu, y_cpu, 1)
    z1_2d = np.polyfit(x_c_cpu, y_cpu, 2)
    y_cpu_predict = np.array(x_c_cpu)*z1[0] + (z1[1])

    

    x_gpu = np.arange(1024,10240,1000,dtype=int)
    x_c_gpu = []
    y_gpu = []
    X_gpu = []
    for index, embed_dim in enumerate(x_gpu):
        kv_params = 2*10*embed_dim
        c = get_c(embed_dim=embed_dim,layer=layer)  
        x_c_gpu.append(c)
        gpu_p1 = {'c':c,'h':embed_dim,'t':0}
        start = time.time()
        module,K,V,input = build_module_and_input(embed_dim=embed_dim,ffn_dim=embed_dim*4)
        K = K_list[index]
        V = V_list[index]
        input = input_list[index]
        print("gpu_build_moudle_time:",time.time()-start)
        start = time.time()
        gpu_t1 = run_gpu_nn(module, input, K, V, 2)[2]
        print("gpu_run_time:",time.time()-start)
        gpu_p1['t'] = gpu_t1
        X_gpu.append([2*c/hardware_information['gpu_ap'],2*c/hardware_information['pcie_bw'],2*c/hardware_information['gpu_mem_bw'],1])
        y_gpu.append(gpu_t1)
    z2 = np.polyfit(x_c_gpu[1:], y_gpu[1:], 1)
    z2_2d = np.polyfit(x_c_gpu[1:], y_gpu[1:], 2)

    const_gpu = np.linalg.solve(X_gpu[-5:-1],y_gpu[-5:-1])

    y_gpu_predict = z2[0]*np.array(x_c_gpu) + z2[1]

    embed_dim = 4096
    m1 = 2*const_cpu[0]/hardware_information['cpu_ap']+2*const_cpu[1]/hardware_information['dram_bw']
    b1 = 2*embed_dim/hardware_information['pcie_bw']*const_cpu[2] + const_cpu[3]

    m2 = 2*const_gpu[0]/hardware_information['gpu_ap']+2*const_gpu[1]/hardware_information['pcie_bw'] + 2*const_gpu[2]/hardware_information['gpu_mem_bw']
    # m2 = 2*const_gpu[1]/32 + 2*const_gpu[2]/960
    b2 = const_gpu[3]

    print(find_intersection(m1,b1,m2,b2))
    num_paras, same_time = find_intersection(m1,b1,m2,b2)
    print("params num:", num_paras)
    avg_paras += np.abs(num_paras)
avg_paras = avg_paras/epochs
print("rand:",rand,"avg_paras:",avg_paras)



print('Analyze done!')
