--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'convnet'

return function(args)
    args.n_units        = {32, 64, 64}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {512}
    args.nl             = nn.Rectifier

    net = create_network(args)
    return net
    
    if args.teacher_net ~= nil then
      print(string.format("copying teacher net %s\n",args.teacher_net))
      teacher_net = torch.load(args.teacher_net)
      for i=1,182 do
        --a:narrow(1,i,1):fill(teacher_net.reward_history[i])
        print(teacher_net.reward_history[i])
      end
      for layer=args.first_layer, args.last_layer do
        print(string.format("copying layer %d\n",layer))
        net.model.modules[layer].weights = teacher_net.model.modules[layer].weights
        net.model.modules[layer].bias = teacher_net.model.modules[layer].bias
      end
    end

    return net
end
