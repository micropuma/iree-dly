#include <fstream>
#include <iostream>

int
main(void) {
  std::ifstream in("/tmp/a", std::ios::in);
  in.seekg(0, std::ifstream::beg);
  in.close();

  return 0;
}
