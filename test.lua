require 'nnbhwd'

ip=16
op=32
kw=5
kh=5
dw=1
dh=1
-- paddings = 0

mpkw=3
mpkh=3
mpdw=2
mpdh=2

iw=25
ih=25
bs=16



if true then 
   if true then 

      torch.setdefaulttensortype('torch.FloatTensor')

      s1=nn.Sequential()
      s2=nn.Sequential()

      foo1=nn.SpatialConvolutionMM(ip,op,kw,kh,dw,dh)
      m1 = nn.SpatialMaxPooling(mpkw,mpkh,dw,dh)
      foo2=nn.SpatialConvolutionBHWD(ip,op,kw,kh,dw,dh)
      m2 = nn.SpatialMaxPoolingBHWD(mpkw,mpkh,dw,dh)

      x2=torch.Tensor(bs,ih,iw,ip):uniform();
      x1=x2:transpose(3,4):transpose(2,3):contiguous();
      x2=x2:cuda()

      foo2.bias:copy(foo1.bias);
      tw=foo1.weight:clone()
      tw:resize(op, ip, kh, kw);
      tw=tw:transpose(2,3):transpose(3,4):contiguous()
      foo2.weight:copy(tw);

      s1:add(foo1)
      s1:add(m1)
      s2:add(foo2)
      s2:add(m2)

      s2:cuda()

      o1=s1:forward(x1):float();
      o2=s2:forward(x2):float();

      t=o2:transpose(3,4):transpose(2,3) - o1;
      print('GPU/float output diff : '..t:abs():max())

      foo1:zeroGradParameters()
      foo2:zeroGradParameters() 

      gradout1=o1:clone():uniform()
      gradout2=gradout1:clone():transpose(2,3):transpose(3,4):contiguous():cuda()

      s1:backward(x1, gradout1);
      s2:backward(x2, gradout2);


      t=s2.gradInput:float():transpose(3,4):transpose(2,3) - s1.gradInput:float();
      print('GPU/float gradInput diff : '..t:abs():max())

      t=foo2.gradWeight:float():transpose(3,4):transpose(2,3) - foo1.gradWeight:float();
      print('GPU/float gradWeight diff : '..t:abs():max())

      t=foo2.gradBias:float() - foo1.gradBias:float();
      print('GPU/float gradBias diff : '..t:abs():max())

   end



   if true then 

      torch.setdefaulttensortype('torch.DoubleTensor')

      s1=nn.Sequential()
      s2=nn.Sequential()

      foo1=nn.SpatialConvolution(ip,op,kw,kh,dw,dh)
      m1 = nn.SpatialMaxPooling(mpkw,mpkh,dw,dh)
      foo2=nn.SpatialConvolutionBHWD(ip,op,kw,kh,dw,dh)
      m2 = nn.SpatialMaxPoolingBHWD(mpkw,mpkh,dw,dh)

      bs=32

      x2=torch.Tensor(bs,ih,iw,ip):uniform();
      x1=x2:transpose(3,4):transpose(2,3):contiguous();

      foo2.bias:copy(foo1.bias);
      foo2.weight:copy(foo1.weight:transpose(2,3):transpose(3,4));

      s1:add(foo1)
      s1:add(m1)
      s2:add(foo2)
      s2:add(m2)


      o1=s1:forward(x1);
      o2=s2:forward(x2);

      t=o2:transpose(3,4):transpose(2,3) - o1
      print('CPU double output diff : '..t:abs():max())

      foo1:zeroGradParameters()
      foo2:zeroGradParameters()

      gradout1=o1:clone():uniform()
      gradout2=gradout1:clone():transpose(2,3):transpose(3,4):contiguous()

      s1:backward(x1, gradout1);
      s2:backward(x2, gradout2);


      t=s2.gradInput:transpose(3,4):transpose(2,3) - s1.gradInput;
      print('CPU double gradInput diff : '..t:abs():max())

      t=foo2.gradWeight:transpose(3,4):transpose(2,3) - foo1.gradWeight;
      print('CPU double gradWeight diff : '..t:abs():max())

      t=foo2.gradBias - foo1.gradBias;
      print('CPU double gradBias diff : '..t:abs():max())

   end
end

