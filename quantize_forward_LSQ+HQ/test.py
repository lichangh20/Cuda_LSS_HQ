import torch
import qmatmul
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import quantize_forward_easy

class MatrixConfig:
    def __init__(self):
        self.M = 4096
        self.K = 4096
        self.N = 4096
        self.testTurn = 30
        self.num_bits = 4
        self.group_size = 32
        
mconfig = MatrixConfig()

T = {}

size = 1
H = torch.ones(1, 1).cuda()
T[1] = H

for i in range(10):
    H = torch.cat((torch.cat([H, H], 1),
                   torch.cat([H, -H], 1)), 0) / math.sqrt(2)
    size *= 2
    T[size] = H


matrix_shape = []
cuda_tflops = []
cuda_hadmard_time = []
cuda_quantize1_time = []
cuda_quantize1_flops = []
cuda_quantize2_time = []
cuda_pack_time = []
cuda_gemm_time = []
cuda_dequantize_time = []
python_ordgemm_flops = []

class PreconditionerTest:
    def __init__(self):
        self.x = torch.randn(mconfig.M, mconfig.K).cuda().half()
        self.y = torch.randn(mconfig.N, mconfig.K).cuda().half()
        self.num_bins = 2 ** mconfig.num_bits - 1
        # self.hadmard = T[mconfig.group_size].repeat(mconfig.K // mconfig.group_size, 1)
        self.scale_hx = 0
        self.scale_hy = 0
        
    # x corresponds to input, y corresponds to weight
    # step1: hadamard quantize input and weight
    # step2: LSQ forward quantize input and weight, with scale_input and scale_weight
    # step3: linear forward input.matmul(weight.t()) and return the result
    def HadmardQuantize_python(self, input, weight):
        hadmard = T[mconfig.group_size].half()
        input_shape = input.shape
        input_batch = input.view(-1,mconfig.group_size)
        h_input = input_batch.matmul(hadmard).view(input_shape)
        
        weight_shape = weight.shape
        weight_batch = weight.view(-1,mconfig.group_size)
        h_weight = weight_batch.matmul(hadmard).view(weight_shape)
        
        self.scale_hx = max(abs(h_input.max()), abs(h_input.min())) / 7
        self.scale_hy = max(abs(h_weight.max()), abs(h_weight.min())) / 7
        
        total_time = 0
        for i in range(mconfig.testTurn + 1):
            # step1: hadamard quantize input and weight
            time1 = time.time()
            h_input = input_batch.matmul(hadmard).view(input_shape)
            h_weight = weight_batch.matmul(hadmard).view(weight_shape)
            
            # step2: LSQ forward quantize input and weight, with scale_input and scale_weight
            matrix1 = (h_input / self.scale_hx).round_().clamp(-8, 7)
            matrix2 = (h_weight / self.scale_hy).round_().clamp(-8, 7)
            # grad_scale_input = 1.0 / math.sqrt(input.numel() * 7)
            # grad_scale_weight = 1.0 / math.sqrt(weight.numel() * 7)
        
            out = self.scale_hx * self.scale_hy * matrix1.matmul(matrix2.t())
            torch.cuda.synchronize()
            time2 = time.time()
            if i >= 1:
                total_time += time2 - time1
        print("HQ python MM speed:")
        # print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        print()
        print("final output is:")
        print(out)

    def Gemm_ordinary_python(self, x, y):
        total_time = 0
        for i in range(mconfig.testTurn+1):
            time1 = time.time()
            out = x.matmul(y.t())
            torch.cuda.synchronize()
            time2 = time.time()
            if i>= 1:
                total_time += time2 - time1
        print("fp16 gemm speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        # print("    result:",out)
        print()
        
        python_ordgemm_flops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
    
    
    def HadmardQuantize_cuda_speed(self, x, y):
        hadmard_time = 0
        quantize1_time = 0
        quantize2_time = 0
        pack_time = 0
        gemm_time = 0
        dequantize_time = 0
        
        hadmard = T[mconfig.group_size].half()
        x_shape = x.shape
        x_batch = x.view(-1,mconfig.group_size)
        y_shape = y.shape
        y_batch = y.view(-1,mconfig.group_size)
        
        total_time = 0
        for i in range(mconfig.testTurn+1):
            time1 = time.time()
            h_x = x_batch.matmul(hadmard).view(x_shape)
            h_y = y_batch.matmul(hadmard).view(y_shape)
            qmatmul.synchronize()
            time_flag = time.time()
            # self.scale_hx = torch.tensor(self.scale_hx)
            out2 = quantize_forward_easy.quantize(h_x,h_y,self.scale_hx, self.scale_hy)
            qmatmul.synchronize()
            if i>= 1:
                hadmard_time += time_flag - time1
                quantize1_time += out2[1][0]
                quantize2_time += out2[1][1]
                pack_time += out2[1][2]
                gemm_time += out2[1][3]
                dequantize_time += out2[1][4]
                time2 = time.time()
                total_time += time2 - time1
        print("HQ cuda MM speed:")
        # print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        print("    output is:")
        print(out2[0])
        print()
        cuda_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        cuda_hadmard_time.append(hadmard_time)
        cuda_quantize1_time.append(quantize1_time)
        cuda_quantize1_flops.append(1e-12 * mconfig.M * mconfig.K * mconfig.testTurn * 2 / quantize1_time)
        cuda_quantize2_time.append(quantize2_time)
        cuda_pack_time.append(pack_time)
        cuda_gemm_time.append(gemm_time)
        cuda_dequantize_time.append(dequantize_time)        
    
def draw_picture_full():
    plt.figure(figsize=(20, 20))
    area = plt.subplot2grid((11,11),(0,0),rowspan=11, colspan = 10)
    area.plot()
    
    bar_width = 0.6
    
    data = [cuda_hadmard_time, np.array(cuda_quantize1_time) + np.array(cuda_quantize2_time), cuda_pack_time, cuda_gemm_time, cuda_dequantize_time]
    labels = ["hadamard", "quantize", "pack", "gemm", "dequantize"]
        
    r1 = range(len(matrix_shape))
    
    bottom_y = np.zeros(len(matrix_shape))
    data = np.array(data)
    sums = np.sum(data, axis=0)
    
    for index in range(len(data)):
        y = data[index] / sums
        plt.bar(r1, y, width=bar_width, edgecolor='white', label=labels[index], bottom=bottom_y)
        bottom_y += y
        
    for i in range(data.shape[1]):
        tmp_y = 0
        for j in range(data.shape[0]):
            y = data[j][i] / sums[i]
            text = "%d%%" % (100 * y)
            plt.text(r1[i], tmp_y+y/2,text,color='white',size='36',horizontalalignment='center',verticalalignment='center')
            tmp_y += y
    
    plt.xticks(r1, matrix_shape, rotation=30, fontsize=30)
    plt.yticks(fontsize=60)

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, fontsize=45)

    plt.ylabel('Time ratio', fontdict={'size' : 60})
    plt.xlabel("Matrix size (M,N,K)", fontdict={'size' : 60})

    plt.savefig('./image/plot_time.pdf', bbox_inches='tight')
    
if __name__=="__main__":
    
    # for (m,n,k) in [(4608, 5120, 6144),(5120,6144,8192),(6144,6144,9216),(7168,6656,8704),(8192,7680,9728),(15360,8704,10752)]:
    for (m,n,k) in [(4608, 5120, 6144)]:
            print("matrix multiplication of {M,N,K} = {%d, %d, %d}" % (m,n,k))
            mconfig.M = m
            mconfig.N = n
            mconfig.K = k
            matrix_shape.append((mconfig.M, mconfig.N, mconfig.K))
            
            test = PreconditionerTest()
            test.HadmardQuantize_python(test.x,test.y)
            test.HadmardQuantize_cuda_speed(test.x, test.y)
            # test.Gemm_ordinary_python(test.x, test.y)
    
    # draw_picture_full()