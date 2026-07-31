#include <vector>
#include <cstddef>


class Tensor {
private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

    void calculate_strides();

    std::size_t flatten_index(std::vector<std::size_t> indices) const;

public:


    Tensor(const std::initializer_list<std::size_t>& shape, float value);

    float& operator() (const std::initializer_list<std::size_t>& indices);

    const std::vector<std::size_t>& get_shape();



};