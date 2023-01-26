import torch
import quantize_grad_weight_speed
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
cuda_easy_tflops = []
twolayer_cuda_speed_tflops = []
hadmard_cuda_speed_tflops = []
cuda_hadmard_time = []
cuda_quantize1_time = []
cuda_quantize2_time = []
cuda_leverage_time = []
cuda_sample_time = []
cuda_pack_time = []
cuda_gemm1_time = []
cuda_gemm2_time = []
cuda_dequantize_time = []
python_ordgemm_flops = []

class PreconditionerTest:
    def __init__(self):
        self.x = torch.randn(mconfig.K, mconfig.M).cuda().half()
        self.y = torch.randn(mconfig.K, mconfig.N).cuda().half()
        self.num_bins = 2 ** mconfig.num_bits - 1
        self.scale_y = max(abs(self.y.min()), abs(self.y.max())) / 7
        self.quantize_y = self.y / self.scale_y
        self.quantize_y.clamp_(-8.0, self.num_bins-8).round_()
        self.quantize_y = self.quantize_y.to(torch.int8)
        self.zero_point1 = 0
        self.scale1 = 0
        self.zero_point2 = 0
        self.scale2 = 0
        self.hadmard = T[mconfig.group_size].half()
        
    def TwoLayerQuantizeWeight_cuda_speed(self, input):
        total_time = 0
        hadmard_time = 0
        quantize1_time = 0
        quantize2_time = 0
        leverage_time = 0
        sample_time = 0
        pack_time = 0
        gemm1_time = 0
        gemm2_time = 0
        dequantize_time = 0
        
        y_shape = self.y.shape
        y_batch = self.y.view(-1,mconfig.group_size)
        
        mn = min(input.min() - 1e-8, 0)
        mx = max(input.max() + 1e-8, 0)
        
        for i in range(mconfig.testTurn + 1):
            time1 = time.time()
            
            hy = y_batch.matmul(self.hadmard).view(y_shape)
            qmatmul.synchronize()
            time2 = time.time()

            second_transform = quantize_grad_weight_speed.quantize(input, self.quantize_y, self.scale_y, hy, self.num_bins)
            qmatmul.synchronize()
            time3 = time.time()
            out_shape = second_transform[0].shape
            output = second_transform[0].view(-1,mconfig.group_size).matmul(self.hadmard).view(out_shape)
            qmatmul.synchronize()
            time4 = time.time()
            if i >= 1:
                total_time += time4 - time1
                hadmard_time += (time2 - time1) + (time4 - time3)
                quantize1_time += second_transform[1][0]
                quantize2_time += second_transform[1][1] 
                leverage_time += second_transform[1][2]
                sample_time += second_transform[1][3]
                pack_time += second_transform[1][4]
                gemm1_time += second_transform[1][5]
                gemm2_time += second_transform[1][6]
                dequantize_time += second_transform[1][7]
                
        print("LSS cuda MM speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        # print("output is:")
        # print(output)
        print()
        twolayer_cuda_speed_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        cuda_hadmard_time.append(hadmard_time)
        cuda_quantize1_time.append(quantize1_time)
        cuda_quantize2_time.append(quantize2_time)
        cuda_leverage_time.append(leverage_time)
        cuda_sample_time.append(sample_time)
        cuda_pack_time.append(pack_time)
        cuda_gemm1_time.append(gemm1_time)
        cuda_gemm2_time.append(gemm2_time)
        cuda_dequantize_time.append(dequantize_time)
        
    def Gemm_ordinary_python(self, x, y):
        total_time = 0
        for i in range(mconfig.testTurn+1):
            time1 = time.time()
            out = x.t().matmul(y)
            torch.cuda.synchronize()
            time2 = time.time()
            if i>= 1:
                total_time += time2 - time1
        print("fp16 gemm speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
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
        h_x = x_batch.matmul(hadmard).view(x_shape)
        h_y = y_batch.matmul(hadmard).view(y_shape)
        scale_hx = max(abs(h_x.max()), abs(h_x.min())) / 7
        scale_hy = max(abs(h_y.max()), abs(h_y.min())) / 7
        for i in range(mconfig.testTurn+1):
            time1 = time.time()
            h_x = x_batch.matmul(hadmard).view(x_shape)
            h_y = y_batch.matmul(hadmard).view(y_shape)
            qmatmul.synchronize()
            time_flag = time.time()
            out2 = quantize_forward_easy.quantize(h_x,h_y,scale_hx, scale_hy)
            qmatmul.synchronize()
            time2 = time.time()
            if i>= 1:
                hadmard_time += time_flag - time1
                quantize1_time += out2[1][0]
                quantize2_time += out2[1][1]
                pack_time += out2[1][2]
                gemm_time += out2[1][3]
                dequantize_time += out2[1][4]                
                total_time += time2 - time1
        print("HQ cuda MM speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        # print("    output is:", out2[0])
        print()
        hadmard_cuda_speed_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        
    def TwoLayerQuantizeWeight_python(self, input):
        # backward quantize x, easy minimax quantize y
        total_time = 0
        actual_time = 0
        for i in range(mconfig.testTurn + 1):
            time1 = time.time()
            mn = min(input.min() - 1e-8, 0)
            mx = max(input.max() + 1e-8, 0)
            
            self.zero_point1 = mn
            self.scale1 = self.num_bins / (mx - mn)
            
            qzero = -self.zero_point1 * self.scale1
            iqzero = torch.floor(qzero)
            
            if iqzero > 0:
                mx = (iqzero - self.num_bins) * mn / iqzero
            elif iqzero == 0:
                self.zero_point1, mn = 0, 0

            self.scale1 = self.num_bins / (mx - mn)
            
            first_transform = (input - self.zero_point1) * self.scale1 - 8
            first_transform.clamp_(-8.0, self.num_bins-8).round_()
            first_quantize = (first_transform+8) / self.scale1 + self.zero_point1
            
            residual = input - first_quantize
            
            mn = min(residual.min() - 1e-8, 0)
            mx = max(residual.max() + 1e-8, 0)
            
            self.zero_point2 = mn
            self.scale2 = self.num_bins / (mx - mn)
            
            qzero = -self.zero_point2 * self.scale2
            iqzero = torch.floor(qzero)
                    
            if iqzero > 0:
                mx = (iqzero - self.num_bins) * mn / iqzero
            elif iqzero == 0:
                self.zero_point2, mn = 0, 0
            self.scale2 = self.num_bins / (mx - mn)
            second_transform = (residual - self.zero_point2) * self.scale2 - 8
            noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
            # second_transform.add_(noise)
            second_transform.clamp_(-8.0, self.num_bins-8).round_()
            second_quantize = (second_transform+8) / self.scale2 + self.zero_point2
            output = torch.cat([first_transform, second_transform], dim=0)
            output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
            
            # leverage score
            y_shape = self.y.shape
            hy = self.y.view(-1,mconfig.group_size).matmul(self.hadmard).view(y_shape)
            y2 = torch.cat([hy, hy], 0)
            x_len = torch.linalg.norm(output_dequantize, dim=1)
            y_len = torch.linalg.norm(y2, dim=1)
            vec_norm = x_len.mul(y_len)
            
            # sampling
            quart_size = int(output.size(0) / 4)
            output_list = torch.split(output, [quart_size, quart_size, quart_size, quart_size], 0)
            sample_x = torch.cat([output_list[0], output_list[2]], 0)
            half_size = int(self.quantize_y.size(0) / 2)
            half_list = torch.split(self.quantize_y, [half_size, half_size], 0)
            sample_y = torch.cat([half_list[0], half_list[0]],0)
            
            # dequantize inputx
            half_shape = int(sample_x.shape[0] / 2)
            first, second = torch.split(sample_x,[half_shape, half_shape], dim=0)
            first = (first+8) / self.scale1 + self.zero_point1
            second = (second+8) / self.scale2 + self.zero_point2
            dequantize_sample_x = torch.cat([first, second], dim=0)
            
            # dequantize inputy
            dequantize_sample_y = sample_y / self.scale_y 
            h_output = dequantize_sample_x.t().matmul(dequantize_sample_y)
            out_shape = h_output.shape
            output = h_output.view(-1,mconfig.group_size).matmul(self.hadmard).view(out_shape)
            
            torch.cuda.synchronize()
            time2 = time.time()
            actual_time += time2 - time1
            if i >= 1:
                total_time += time2 - time1
            
        print("quantize python:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        print()
        # print("final output is:")
        # print(output)
         
def draw_picture_flops():
    plt.figure(figsize=(25, 20))
    bar_width = 0.25
    
    r1 = range(len(matrix_shape))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    plt.bar(r1, python_ordgemm_flops, width=bar_width, edgecolor='white', label='fp16')
    plt.bar(r2, twolayer_cuda_speed_tflops, width=bar_width, edgecolor='white', label='LSS')
    plt.bar(r3, hadmard_cuda_speed_tflops, width=bar_width, edgecolor='white', label='HQ')
    
    plt.xticks(r2, matrix_shape, rotation=30, fontsize=30)
    plt.yticks(fontsize=60)
    
    plt.legend(loc='upper left', fontsize=60)

    font = {'size' : 60}
    plt.xlabel("Matrix size (M,N,K)",font)
    plt.ylabel('Tflops', font)
    # plt.title('Comparison of FP16 MM, HQ and LSS operator',fontsize=60)
    
    plt.savefig('./image/plot_flops.pdf', bbox_inches='tight')
    
def draw_picture_full():
    plt.figure(figsize=(20, 20))
    area = plt.subplot2grid((11,11),(0,0),rowspan=11, colspan = 10)
    area.plot()
    bar_width = 0.6
    
    data = [cuda_hadmard_time, np.array(cuda_quantize1_time) , np.array(cuda_quantize2_time), cuda_leverage_time, cuda_sample_time, cuda_pack_time, np.array(cuda_gemm1_time) + np.array(cuda_gemm2_time),cuda_dequantize_time]
    labels = ["hadamard", "quantize1", "quantize2","leverage", "sample", "pack", "gemm", "dequantize"]
        
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
            plt.text(r1[i], tmp_y+y/2, text, color='white',size='36',horizontalalignment='center',verticalalignment='center')
            tmp_y += y
    
    plt.xticks(r1, matrix_shape, rotation=30, fontsize=30)
    plt.yticks(fontsize=60)

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, fontsize=45)

    plt.ylabel('Time ratio', fontdict={'size' : 60})
    plt.xlabel("Matrix size (M,N,K)", fontdict={'size' : 60})

    plt.savefig('./image/plot_time.pdf', bbox_inches='tight')
    
    
if __name__=="__main__":
    for (m,n,k) in [(4608, 5120, 6144),(5120,6144,8192),(6144,6144,9216),(7168,6656,8704),(8192,7680,9728),(15360,8704,10752)]:
        print("matrix multiplication of {M,N,K} = {%d, %d, %d}" % (m,n,k))
        mconfig.M = m
        mconfig.N = n
        mconfig.K = k
        matrix_shape.append((mconfig.M, mconfig.N, mconfig.K))
        test = PreconditionerTest()
        test.TwoLayerQuantizeWeight_cuda_speed(test.x)
        test.Gemm_ordinary_python(test.x, test.y)
        test.HadmardQuantize_cuda_speed(test.x.t().contiguous(), test.y.t().contiguous())

    draw_picture_flops()
    draw_picture_full()