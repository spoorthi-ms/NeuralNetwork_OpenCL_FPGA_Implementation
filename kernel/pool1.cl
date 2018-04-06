**********************************************************************************************************
DESCRIPTION: Kernel code for Pool Layer 1 of Cifarnet.
             Consists of: 3 Convolution Layers, 2 Fully Connected Layers, 3 Pool Layers.

			This code is developed in OpenCL using Vivado HLS. 
			Bitstream is generated using Vivado.
			The host code is developed in Xilinx SDK.


Author: Spoorthi Mysore Shivakumar (shivakumar.spoorthi@gmail.com)

**********************************************************************************************************
__kernel void __attribute__ ((reqd_work_group_size(1024,1,1))) p1L(__global float *Layer1_Features, __global float *Layer2_pool_GPU)
		{
	int z = get_global_id(0);
	int row = z/32;
	int col = z%32;

	float max = 0;
	{
		for(int output =0;output < 32 ;output++)
		{
			if(row%2 != 0)
			{
				if(col%2 != 0)
				{
					for(int i = row-1; i <= row+1; i++)
					{
						if(i>31) break;
						for(int j = col-1; j <= col+1; j++)
						{
							if(j>31) break;
							if(max < ((Layer1_Features[output*32*32+i*32+j])))
								max =   ((Layer1_Features[output*32*32+i*32+j])) ;

						}
					}
					//ReLU activation function compuation
					if(max<0)
						max = 0;
					Layer2_pool_GPU[output*16*16+(row-1)*8+(col-1)/2] = max;
					max = 0;
				}
			}
		}
	}
}
