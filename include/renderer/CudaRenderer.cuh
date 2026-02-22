
        #pragma once
        #include "renderer/Renderer.hpp"
        
        // A minimal, non-CUDA declaration of CudaRenderer for macOS to compile.
        class CudaRenderer : public Renderer {
        public:
            CudaRenderer();
            ~CudaRenderer();
            void Draw(const Character& character, unsigned int frameCount) override;
        };
        