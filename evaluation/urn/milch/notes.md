
TODO: install armadillo (14.6. for c++11)

export CPATH=/opt/homebrew/include
export LIBRARY_PATH=/opt/homebrew/lib

to install from source
sudo install_name_tool -id /usr/local/lib/libarmadillo.14.dylib /usr/local/lib/libarmadillo.14.dylib

```
cd swift
make compile
```

```
./swift -i ../urn_biased.blog -o src/urn_biased.cpp -e LWSampler -n 10000000
```

```
cd ..
python3 patch.py
```

```
cd swift/src
g++ -o urn_biased.out -std=c++11 -O3 urn_biased.cpp random/*.cpp -larmadillo
```

```
mv urn_biased.out ../../urn_biased.out
```

```
cd ../..
./urn_biased.out
```