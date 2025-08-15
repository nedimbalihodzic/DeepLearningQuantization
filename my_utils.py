import torch

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype = torch.int8):
    scaled_and_shifted_tesor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tesor)
    
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    return q_tensor


def linear_dequantization(quantized_tensor, scale, zero_point):
    return scale * (quantized_tensor.float() - zero_point)


def get_q_scale_and_zero_point(tensor, dtype = torch.int8):
    q_min = torch.iinfo(torch.int8).min
    q_max = torch.iinfo(torch.int8).max
    r_min = tensor.min().item()
    r_max = tensor.max().item()
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min/scale)
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max 
    else: 
        zero_point = int (round(zero_point)) 
    
    return scale, zero_point 



def linear_quantization(tensor, dytpe=torch.int8):
    scale, zero_point = get_q_scale_and_zero_point(tensor,dtype=dytpe)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=dytpe)
    return quantized_tensor, scale, zero_point

def get_q_scale_symmetric(tensor, dtype = torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max
    return r_max / q_max

def linear_q_symmetric(tensor, dtype = torch.int8):
    scale = get_q_scale_symmetric(tensor)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale = scale, zero_point=0, dtype=dtype)
    return quantized_tensor, scale 


def linear_q_symmetric_per_channel(r_tensor, dim, dtype = torch.int8):

    output_dim = r_tensor.shape[dim]
    # store the scales
    scale = torch.zeros(output_dim)

    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim,index)
        scale[index] = get_q_scale_symmetric(sub_tensor,dtype=dtype)
    
    # reshape the scale 
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1 
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(r_tensor, scale = scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale


def linear_q_symmetric_per_group(tensor, group_size, dtype = torch.int8):
    t_shape = tensor.shape
    # we need to make sure that each row is divisible by group size 
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2
    tensor = tensor.view(-1, group_size)
    quantized_tensor, scale = linear_q_symmetric_per_channel(tensor, dim=0, dtype=dtype)
    quantized_tensor = quantized_tensor.view(t_shape)
    return quantized_tensor, scale  


def linear_dequantization_per_group(quantized_tensor, scale, group_size):
    q_shape = quantized_tensor.shape 
    quantized_tensor = quantized_tensor.view(-1,group_size)
    dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)
    dequantized_tensor = dequantized_tensor.view(q_shape)
    return dequantized_tensor
