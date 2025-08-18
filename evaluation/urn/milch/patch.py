
with open("swift/src/util/Hist.h", "r") as f:
    filecontent = f.read()

with open("swift/src/util/Hist.h", "w") as f:
    f.write(filecontent.replace("%d -> %.8lf", "%d -> %.15e"))
    
    
with open("swift/src/random/SwiftDistribution.h", "r") as f:
    filecontent = f.read()

with open("swift/src/random/SwiftDistribution.h", "w") as f:
    f.write(filecontent.replace("protected:", ""))
    

original_sample = """
void BLOGProgram::sample(int n)
{
  double __local_weight;
  init(n);
  for (_cur_loop=1;_cur_loop<n;_cur_loop++)
  {
    if (_set_evidence(__local_weight))
      _evaluate_query(__local_weight);
    _weight[_cur_loop]=__local_weight;
  }

}
"""
patched_sample = """
void BLOGProgram::sample(int n)
{
  double __local_weight;
  double __max_weight = -INFINITY;
  init(n);
  for (_cur_loop=1;_cur_loop<n;_cur_loop++)
  {
    if (_set_evidence(__local_weight))
      _evaluate_query(__local_weight);
    _weight[_cur_loop]=__local_weight;
    __max_weight = fmax(__max_weight,__local_weight);
  }
  double __log_Z = 0.;
  for (_cur_loop=1;_cur_loop<n;_cur_loop++)
  {
    __log_Z += exp(_weight[_cur_loop] - __max_weight);
  }
  __log_Z = log(__log_Z) + __max_weight - log((double) n);
  printf("log_Z = %.15e\\n", __log_Z);
}
"""
  
  
orig_main = """
int main()
{
"""

patched_main = """
int main(int argc, char* argv[])
{
  unsigned int seed = 0;
  if (argc > 1) {
      seed = std::stoul(argv[1]);
      std::cout << "Using seed " << seed << std::endl;
  } else {
      std::cout << "No seed provided, using default 0" << std::endl;
  }
  SwiftDistribution<int>::engine.seed(seed);
"""
  
orig_header = """
#include "util/DynamicTable.h"

using namespace std;
using namespace swift::random;


int main();
"""

patched_header = """
#include "util/DynamicTable.h"
#include "random/SwiftDistribution.h"

using namespace std;
using namespace swift::random;


int main(int argc, char* argv[]);
"""

for file in ["swift/src/urn_biased.cpp", "swift/src/urn_dirac.cpp"]:
    with open(file, "r") as f:
        filecontent = f.read()
    with open(file, "w") as f:
        f.write(
          filecontent.replace(original_sample, patched_sample).replace(orig_main, patched_main).replace(orig_header, patched_header)
        )