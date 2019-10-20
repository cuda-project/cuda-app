git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc
cd grpc
git submodule update --init
make
sudo make install

cd /Users/tonye/CLionProjects/CLionProjects-cuda/cuda-app/grpcapp
protoc --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` helloworld.proto
protoc --cpp_out=. helloworld.proto