#ifdef __APPLE__
#import <MetalKit/MetalKit.h>

@interface GameView : MTKView

- (void)setMouseLock:(BOOL)locked;

@end


#endif