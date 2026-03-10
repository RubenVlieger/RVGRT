#include "renderer/Renderer.hpp"
        class CudaRenderer : public Renderer {
        public:
            CudaRenderer() {}
            ~CudaRenderer() {}
            void Draw(const Character&, unsigned int) override {}
        };
