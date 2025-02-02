util.global mutable @a : i32
func.func @fool() {
  %c5 = arith.constant 5 : i32
  util.global.store %c5, @a : i32
  return
}
