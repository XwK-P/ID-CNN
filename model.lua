require 'nngraph'
require 'TotalVariation'

function defineGour_net(input_nc, output_nc, ngf)
   -- e0 = - nn.Tanh()

    e0 = - nn.Identity()
    e1 = e0 - nn.SpatialConvolution(input_nc, ngf, 3, 3, 1, 1, 1, 1)
    --
    e2 = e1 -  nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    --
    e3 = e2 -  nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    --
    e4 = e3 -  nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    --
    d1 = e4 -  nn.LeakyReLU(0.2, true) - nn.SpatialFullConvolution(ngf, ngf, 3, 3,1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    --
    --d1 = {d1_,e3} - nn.CAddTable(true)
    --
    d2= d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, ngf, 3, 3,1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    --
    --d2 = {d2_,e2} - nn.CAddTable(true)
    --
    d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    --
    --d3 = {d3_,e1} - nn.CAddTable(true)
    --
    d6 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 3, 3, 1, 1, 1, 1)
    --d6 = d61 - nn.Identity()

    --
    
    --
    o1 = d6 -  nn.ReLU(true)

    -- o1_ = {o1,o1} - nn.CDivTable(true)


    o2 = o1 - nn.AddConstant(1e-7)

    o3 = {e0,o2} - nn.CDivTable(true)

    --o4 = o3 - nn.SpatialConvolution(output_nc, output_nc, 3, 3, 1, 1, 1, 1) - nn.Tanh()
    o4 = o3  - nn.Tanh()  - nn.TotalVariation(1e-5)

    
    netG = nn.gModule({e0},{o4})

    graph.dot(netG.fg,'netG','mynetG')
      
    return netG
end
