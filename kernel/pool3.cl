**********************************************************************************************************
DESCRIPTION: Kernel code for Pool Layer 3 of Cifarnet.
             Consists of: 3 Convolution Layers, 2 Fully Connected Layers, 3 Pool Layers.

			This code is developed in OpenCL using Vivado HLS. 
			Bitstream is generated using Vivado.
			The host code is developed in Xilinx SDK.


Author: Spoorthi Mysore Shivakumar (shivakumar.spoorthi@gmail.com)

**********************************************************************************************************
__kernel void __attribute__ ((reqd_work_group_size(64,1,1))) p3(__global float *Layer3_Neurons_GPU, __global float *Layer3_pool_GPU)
		{

	int z = get_global_id(0);
	int row = z/8;
	int col = z%8;

	float avg = 0;
	int count = 0;

	{
		for(int output =0; output < 64 ; output++)
		{
			if((row%2 != 0) && (row<8))
			{
				if((col%2 != 0) && (col<8))
				{
					for(int i = row-1; i <= row+1; i++)
					{
						if(i>7) break;
						for(int j = col-1; j <= col+1; j++)
						{
							if(j>7) break;
							avg+= ((Layer3_Neurons_GPU[output*8*8 + i*8 + j]));
							count++;

						}
					}
					Layer3_pool_GPU[output*4*4+(row-1)*2+(col-1)/2] = avg/count;
					avg = 0;
					count=0;
				}
			}
		}
	}
		}
