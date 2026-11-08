#include <cmath>
#include <iostream>
#include "policynav/policy.hpp"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  policynav::PolicyStepper p(argv[1]);
  const auto& m = p.meta();
  std::vector<float> depth(m.depth_h * m.depth_w, 1.0f);
  const auto result = p.step(depth, {1.2, .3, -.3}, {1., 0., 0., 0.}, 1.0);
  const double expected[] = {.24, .10, -.15};
  for (int i = 0; i < 3; ++i) {
    if (std::abs(result.proprio[i] - expected[i]) > 1e-6) {
      std::cerr << "wrong velocity normalization: " << result.proprio[i] << '\n';
      return 1;
    }
  }
  return 0;
}
