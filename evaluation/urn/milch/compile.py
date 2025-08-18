import subprocess
# directory milch

subprocess.run(["./swift", "-i", "../urn_biased.blog", "-o", "src/urn_biased.cpp", "-e", "LWSampler", "-n", "10000000"], cwd="./swift")
subprocess.run(["./swift", "-i", "../urn_dirac.blog", "-o", "src/urn_dirac.cpp", "-e", "LWSampler", "-n", "10000000"], cwd="./swift")


subprocess.run(["python3", "patch.py"])


subprocess.run(["g++ -o urn_biased.out -std=c++11 -O3 swift/src/urn_biased.cpp swift/src/random/*.cpp -larmadillo"], shell=True)
subprocess.run(["g++ -o urn_dirac.out -std=c++11 -O3 swift/src/urn_dirac.cpp swift/src/random/*.cpp -larmadillo"], shell=True)
