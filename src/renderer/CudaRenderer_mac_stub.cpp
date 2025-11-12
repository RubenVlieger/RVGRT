
        #include "renderer/Renderer.hpp"
        class CudaRenderer : public Renderer {
        public:
            CudaRenderer() {}
            ~CudaRenderer() {}
            void Draw(const Character&, unsigned int) override {}
        };
        // This is necessary to satisfy the `new CudaRenderer()` in the Windows code path
        // which, although guarded, might still be seen by some analysis tools.
        // A better approach is a factory function, but this works for now.
        