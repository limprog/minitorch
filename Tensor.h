#include <vector>
#include <cstddef>
#include <memory>
#include <ostream>
#include <type_traits>
#include <functional>
#include <unordered_set>


struct Storage {
    std::vector<float> data_;
};


struct AutogradNode;


class Tensor {
private:
    std::shared_ptr<Storage> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

    std::shared_ptr<AutogradNode> node_;

    bool is_contiguous() const;

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

    std::size_t batch_offset(std::size_t batch_index) const;

    void batched_matmul(const Tensor& other,
        Tensor& result,
        std::size_t rows,
        std::size_t cols,
        std::size_t inner,
        std::size_t batch_count) const;

    void accumulate_grad(const Tensor& grad) const;
    static void accumulate_grad(const std::shared_ptr<AutogradNode>& node, const Tensor& grad);
    static void build_path(
        const std::shared_ptr<AutogradNode>& node,
        std::unordered_set<AutogradNode*>& visited,
        std::vector<std::shared_ptr<AutogradNode>>& path
        );


    Tensor copy_for_operation() const;

public:

    Tensor(const std::initializer_list<std::size_t>& shape, float value = 0.0f, bool requires_grad = true);
    Tensor(const std::vector<std::size_t>& shape, float value = 0.0f, bool requires_grad = true);

    float& operator() (const std::initializer_list<std::size_t>& indices);
    const float& operator()(const std::initializer_list<std::size_t>& indices) const;

    template<typename... Indices>
    float& operator[](Indices... indices) {
       static_assert(
           (std::is_convertible_v<Indices, std::size_t> && ...),
           "Tensor indices must be integer-like"
       );

       return storage_->data_[flatten_index(
           std::vector<std::size_t>{
               static_cast<std::size_t>(indices)...
           }
       )];
    }

    template<typename... Indices>
    const float& operator[](Indices... indices) const {
        static_assert(
            (std::is_convertible_v<Indices, std::size_t> && ...),
            "Tensor indices must be integer-like"
        );

        return storage_->data_[flatten_index(
            std::vector<std::size_t>{
                static_cast<std::size_t>(indices)...
            }
        )];
    };

    const std::vector<std::size_t>& get_shape();

    bool requires_grad() const;
    void backward() const;
    Tensor grad() const;

    Tensor& fill_(float value);

    Tensor& reshape_(std::initializer_list<std::size_t> shape);
    Tensor reshape(std::initializer_list<std::size_t> shape) const;
    Tensor& reshape_(std::vector<std::size_t> shape);
    Tensor reshape(std::vector<std::size_t> shape) const;

    Tensor& unsqueeze_(std::size_t axis);
    Tensor unsqueeze(std::size_t axis) const;
    Tensor& squeeze_(std::size_t axis);
    Tensor squeeze(std::size_t axis) const;

    Tensor transpose(std::size_t dim1, std::size_t dim2) const;
    Tensor T() const;



    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator-() const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    Tensor matmul(const Tensor& other) const;
    Tensor sum() const;
    Tensor mean() const;



    Tensor detach() const;

    Tensor copy() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

};


struct AutogradNode {
    std::optional<Tensor> grad;

    std::vector<std::shared_ptr<AutogradNode>> parents;

    std::function<void(const Tensor&)> backward_fn;
};


Tensor operator*(float scalar, const Tensor& tensor);
