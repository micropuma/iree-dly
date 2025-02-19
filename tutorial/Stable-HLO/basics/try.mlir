%token_wait = dataflow.wait_all async [%token_in] {id = 1 : i32}

%token_exec = dataflow.dispatch_region [%token_wait] -> (memref<64x64xf32, 2>) {
  ...
} {id = 2 : i32}

