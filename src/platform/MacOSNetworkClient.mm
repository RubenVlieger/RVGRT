#include "platform/NetworkClient.hpp"
#include "Character.hpp"
#include <mutex>
#include <iostream>

#ifdef __APPLE__
#import <Foundation/Foundation.h>

// Removed placeholder Create()

class MacOSNetworkClient : public NetworkClient {
private:
    NSURLSession* session;
    NSURLSessionWebSocketTask* webSocketTask;
    int myClientID;
    std::mutex stateMutex;
    std::vector<std::vector<float>> latestForeignStates;
    std::string savedUrl;
    ChatCallback _chatCallback;
    BlockEditCallback _blockEditCallback;
    BlockSyncCallback _blockSyncCallback;
    BlockResetCallback _blockResetCallback;
    
    void ReadLoop() {
        if (!webSocketTask) return;
        
        MacOSNetworkClient* weakThis = this;
        [webSocketTask receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage * _Nullable message, NSError * _Nullable error) {
            if (error) {
                if (error.code != NSURLErrorCancelled) {
                    NSLog(@"WebSocket read error: %@", error);
                    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                        NSLog(@"Reconnection timer fired. Checking valid state...");
                        if (weakThis) {
                            if (!weakThis->savedUrl.empty()) {
                                weakThis->Connect(weakThis->savedUrl);
                            } else {
                                NSLog(@"Error: savedUrl is empty!");
                            }
                        } else {
                            NSLog(@"Error: weakThis is null!");
                        }
                    });
                }
                return;
            }
            
            if (message && message.type == NSURLSessionWebSocketMessageTypeString) {
                NSString* text = message.string;
                NSData* data = [text dataUsingEncoding:NSUTF8StringEncoding];
                NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                
                if (json) {
                    NSString* type = json[@"type"];
                    if ([type isEqualToString:@"init"]) {
                        weakThis->myClientID = [json[@"client_id"] intValue];
                        NSLog(@"NetworkClient: Connected! Assigned Client ID: %d", weakThis->myClientID);
                    }
                    else if ([type isEqualToString:@"broadcast"]) {
                        NSDictionary* players = json[@"players"];
                        std::vector<std::vector<float>> newlyReceived;
                        
                        for (NSString* cidKey in players) {
                            int cid = [cidKey intValue];
                            if (cid == weakThis->myClientID) {
                                continue;
                            }
                            
                            NSDictionary* stateData = players[cidKey];
                            NSArray* transforms = stateData[@"transforms"];
                            if (transforms && [transforms isKindOfClass:[NSArray class]] && transforms.count == 112) {
                                std::vector<float> matrices(112);
                                for (int i = 0; i < 112; i++) {
                                    matrices[i] = [transforms[i] floatValue];
                                }
                                newlyReceived.push_back(matrices);
                            }
                        }
                        
                        std::lock_guard<std::mutex> lock(weakThis->stateMutex);
                        weakThis->latestForeignStates = std::move(newlyReceived);
                    }
                    else if ([type isEqualToString:@"chat"]) {
                        int senderId = [json[@"client_id"] intValue];
                        NSString* senderName = json[@"sender"];
                        NSString* chatText = json[@"text"];
                        if (weakThis->_chatCallback) {
                            std::string sender = senderName ? [senderName UTF8String] : "Unknown";
                            std::string txt = chatText ? [chatText UTF8String] : "";
                            weakThis->_chatCallback(senderId, sender, txt);
                        }
                    }
                    else if ([type isEqualToString:@"block"]) {
                        int32_t bx = [json[@"x"] intValue];
                        int32_t by = [json[@"y"] intValue];
                        int32_t bz = [json[@"z"] intValue];
                        uint8_t matID = (uint8_t)[json[@"mat_id"] intValue];
                        if (weakThis->_blockEditCallback) {
                            weakThis->_blockEditCallback(bx, by, bz, matID);
                        }
                    }
                    else if ([type isEqualToString:@"block_sync"]) {
                        NSArray* changes = json[@"changes"];
                        if (changes && [changes isKindOfClass:[NSArray class]]) {
                            std::vector<BlockEdit> edits;
                            edits.reserve([changes count]);
                            for (NSDictionary* change in changes) {
                                BlockEdit edit;
                                edit.x = [change[@"x"] intValue];
                                edit.y = [change[@"y"] intValue];
                                edit.z = [change[@"z"] intValue];
                                edit.matID = (uint8_t)[change[@"mat_id"] intValue];
                                edits.push_back(edit);
                            }
                            if (weakThis->_blockSyncCallback) {
                                weakThis->_blockSyncCallback(edits);
                            }
                        }
                    }
                    else if ([type isEqualToString:@"block_reset"]) {
                        if (weakThis->_blockResetCallback) {
                            weakThis->_blockResetCallback();
                        }
                    }
                }
            }
            weakThis->ReadLoop();
        }];
    }

public:
    MacOSNetworkClient() : session(nil), webSocketTask(nil), myClientID(-1) {}
    
    ~MacOSNetworkClient() {
        Disconnect();
    }
    
    void Connect(const std::string& url) override {
        if (webSocketTask) {
            Disconnect();
        }
        
        savedUrl = url;
        
        NSLog(@"NetworkClient connecting to %s...", url.c_str());
        
        NSURL* nsUrl = [NSURL URLWithString:[NSString stringWithUTF8String:url.c_str()]];
        NSURLSessionConfiguration* config = [NSURLSessionConfiguration defaultSessionConfiguration];
        session = [NSURLSession sessionWithConfiguration:config];
        
        webSocketTask = [session webSocketTaskWithURL:nsUrl];
        [webSocketTask resume];
        
        // Kick off our read loop
        ReadLoop();
    }
    
    void Disconnect() override {
        if (webSocketTask) {
            [webSocketTask cancelWithCloseCode:NSURLSessionWebSocketCloseCodeNormalClosure reason:nil];
            webSocketTask = nil;
        }
        if (session) {
            [session invalidateAndCancel];
            session = nil;
        }
        myClientID = -1;
    }
    
    void SendState(const Character& localCharacter) override {
        if (!webSocketTask || webSocketTask.state != NSURLSessionTaskStateRunning) {
            return;
        }
        
        // Pack 7 bodily matrices into NSArray
        NSMutableArray* transforms = [NSMutableArray arrayWithCapacity:112];
        
        const float* ptrList[7] = {
            (const float*)&localCharacter.boundingBox.inverseModelMatrix,
            (const float*)&localCharacter.head.inverseModelMatrix,
            (const float*)&localCharacter.trunk.inverseModelMatrix,
            (const float*)&localCharacter.leftArm.inverseModelMatrix,
            (const float*)&localCharacter.rightArm.inverseModelMatrix,
            (const float*)&localCharacter.leftLeg.inverseModelMatrix,
            (const float*)&localCharacter.rightLeg.inverseModelMatrix
        };
        
        for (int p = 0; p < 7; p++) {
            for (int i = 0; i < 16; i++) {
                float val = ptrList[p][i];
                if (std::isnan(val) || std::isinf(val)) {
                    val = 0.0f;
                }
                [transforms addObject:@(val)];
            }
        }
        
        NSDictionary* payload = @{
            @"type": @"state",
            @"data": @{ @"transforms": transforms }
        };
        
        NSError* error = nil;
        NSData* jsonData = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&error];
        if (!error && jsonData) {
            NSString* jsonStr = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
            NSURLSessionWebSocketMessage* msg = [[NSURLSessionWebSocketMessage alloc] initWithString:jsonStr];
            
            [webSocketTask sendMessage:msg completionHandler:^(NSError * _Nullable sendError) {
                if (sendError) {
                    NSLog(@"Error sending state message: %@", sendError);
                }
            }];
        }
    }
    
    void PollUpdates(std::vector<Character>& otherCharacters) override {
        std::lock_guard<std::mutex> lock(stateMutex);
        
        if (latestForeignStates.empty()) {
            return; // Don't wipe 'otherCharacters' instantly if connection blipped
        }
        
        otherCharacters.resize(latestForeignStates.size());
        
        for (size_t c = 0; c < latestForeignStates.size(); c++) {
            const std::vector<float>& matrices = latestForeignStates[c];
            
            float* destPtrs[7] = {
                (float*)&otherCharacters[c].boundingBox.inverseModelMatrix,
                (float*)&otherCharacters[c].head.inverseModelMatrix,
                (float*)&otherCharacters[c].trunk.inverseModelMatrix,
                (float*)&otherCharacters[c].leftArm.inverseModelMatrix,
                (float*)&otherCharacters[c].rightArm.inverseModelMatrix,
                (float*)&otherCharacters[c].leftLeg.inverseModelMatrix,
                (float*)&otherCharacters[c].rightLeg.inverseModelMatrix
            };
            
            int offset = 0;
            for (int p = 0; p < 7; p++) {
                for (int i = 0; i < 16; i++) {
                    destPtrs[p][i] = matrices[offset++];
                }
            }
        }
    }

    void SendChat(const std::string& sender, const std::string& text) override {
        if (!webSocketTask || webSocketTask.state != NSURLSessionTaskStateRunning) {
            return;
        }
        
        NSDictionary* payload = @{
            @"type": @"chat",
            @"sender": [NSString stringWithUTF8String:sender.c_str()],
            @"text": [NSString stringWithUTF8String:text.c_str()]
        };
        
        NSError* error = nil;
        NSData* jsonData = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&error];
        if (!error && jsonData) {
            NSString* jsonStr = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
            NSURLSessionWebSocketMessage* msg = [[NSURLSessionWebSocketMessage alloc] initWithString:jsonStr];
            
            [webSocketTask sendMessage:msg completionHandler:^(NSError * _Nullable sendError) {
                if (sendError) {
                    NSLog(@"Error sending chat message: %@", sendError);
                }
            }];
        }
    }

    void SetChatCallback(ChatCallback callback) override {
        _chatCallback = callback;
    }

    void SendBlockEdit(int32_t x, int32_t y, int32_t z, uint8_t matID) override {
        if (!webSocketTask || webSocketTask.state != NSURLSessionTaskStateRunning) {
            return;
        }

        NSDictionary* payload = @{
            @"type": @"block",
            @"x": @(x),
            @"y": @(y),
            @"z": @(z),
            @"mat_id": @(matID)
        };

        NSError* error = nil;
        NSData* jsonData = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&error];
        if (!error && jsonData) {
            NSString* jsonStr = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
            NSURLSessionWebSocketMessage* msg = [[NSURLSessionWebSocketMessage alloc] initWithString:jsonStr];

            [webSocketTask sendMessage:msg completionHandler:^(NSError * _Nullable sendError) {
                if (sendError) {
                    NSLog(@"Error sending block edit message: %@", sendError);
                }
            }];
        }
    }

    void SetBlockEditCallback(BlockEditCallback callback) override {
        _blockEditCallback = callback;
    }

    void SetBlockSyncCallback(BlockSyncCallback callback) override {
        _blockSyncCallback = callback;
    }

    void SetBlockResetCallback(BlockResetCallback callback) override {
        _blockResetCallback = callback;
    }
};

std::unique_ptr<NetworkClient> NetworkClient::Create() {
    return std::make_unique<MacOSNetworkClient>();
}

#endif // __APPLE__
