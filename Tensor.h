#include <vector>
#include <cstddef>
#include <ostream>


class Tensor {
private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

    void calculate_strides();

    std::size_t flatten_index(std::vector<std::size_t> indices) const;
    void matmul_matrix_kernel(
        const Tensor& other,
        Tensor& result,
        std::size_t offset_a,
        std::size_t offset_b,
        std::size_t offset_result,
        std::size_t rows,
        std::size_t cols,
        std::size_t inner) const;


    void batched_matmul(const Tensor& other,
        Tensor& result,
        std::size_t rows,
        std::size_t cols,
        std::size_t inner,
        std::size_t batch_count) const;

public:

    Tensor(const std::initializer_list<std::size_t>& shape, float value = 0.0f);
    Tensor(const std::vector<std::size_t>& shape, float value = 0.0f);

    float& operator() (const std::initializer_list<std::size_t>& indices);
    const float& operator()(const std::initializer_list<std::size_t>& indices) const;

    const std::vector<std::size_t>& get_shape();

    Tensor& fill_(float value);

    Tensor& reshape_(std::initializer_list<std::size_t> shape);
    Tensor reshape(std::initializer_list<std::size_t> shape) const;

    Tensor& unsqueeze_(std::size_t axis);
    Tensor unsqueeze(std::size_t axis) const;
    Tensor& squeeze_(std::size_t axis);
    Tensor squeeze(std::size_t axis) const;



    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

};
