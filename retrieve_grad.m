clear all;

P = py.sys.path;
if count(P,'get_grad') == 0
    insert(P,int32(0),'get_grad');
end
py.importlib.import_module('get_grad');

out = py.get_grad.get_val_and_grad([1;2;3;4;5;6]);

grad = out{1}{1}
val = out{2}