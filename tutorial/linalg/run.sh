iree-run-module --device=cuda \
  --module=linalg.vmfb \
  --input=10xf32=[0,1,2,3,4,5,6,7,8,9] \
  --input=10xf32=[90,80,70,60,50,40,30,20,10,0]