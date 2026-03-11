#ifdef _WIN32
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0A00
#ifdef WINVER
#undef WINVER
#endif
#define WINVER 0x0A00
#endif

#include "platform/NetworkClient.hpp"
#include "Character.hpp"
#include <mutex>
#include <string>
#include <vector>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>

#pragma comment(lib, "winhttp.lib")

// A microscopic JSON parser just to extract "type":"init" and "transforms":[...] arrays
// Keeps binary size practically zero.
static int ParseClientID(const std::string& json) {
    size_t idPos = json.find("\"client_id\"");
    if (idPos != std::string::npos) {
        size_t colon = json.find(":", idPos);
        if (colon != std::string::npos) {
            return std::stoi(json.substr(colon + 1));
        }
    }
    return -1;
}

static bool ExtractTransforms(const std::string& stateData, std::vector<float>& outFloats) {
    size_t start = stateData.find("\"transforms\"");
    if (start == std::string::npos) return false;
    start = stateData.find("[", start);
    if (start == std::string::npos) return false;
    size_t end = stateData.find("]", start);
    if (end == std::string::npos) return false;
    
    std::string arr = stateData.substr(start + 1, end - start - 1);
    std::stringstream ss(arr);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try { outFloats.push_back(std::stof(item)); } catch (...) {}
    }
    return outFloats.size() == 112;
}

// Removed placeholder Create()

class Win32NetworkClient : public NetworkClient {
private:
    HINTERNET hSession = NULL;
    HINTERNET hConnect = NULL;
    HINTERNET hRequest = NULL;
    HINTERNET hWebSocket = NULL;
    
    int myClientID = -1;
    bool running = false;
    
    std::mutex stateMutex;
    std::vector<std::vector<float>> latestForeignStates;
    HANDLE hThread = NULL;

    static DWORD WINAPI ReceiveThread(LPVOID param) {
        Win32NetworkClient* client = (Win32NetworkClient*)param;
        client->ReadLoop();
        return 0;
    }
    
    void ReadLoop() {
        std::vector<char> buffer(65536);
        DWORD bytesRead = 0;
        WINHTTP_WEB_SOCKET_BUFFER_TYPE bufferType;
        
        while (running && hWebSocket) {
            DWORD error = WinHttpWebSocketReceive(hWebSocket, buffer.data(), buffer.size(), &bytesRead, &bufferType);
            if (error == ERROR_SUCCESS && bytesRead > 0 && bufferType == WINHTTP_WEB_SOCKET_UTF8_MESSAGE_BUFFER_TYPE) {
                std::string text(buffer.data(), bytesRead);
                
                if (text.find("\"type\":\"init\"") != std::string::npos || text.find("\"type\": \"init\"") != std::string::npos) {
                    myClientID = ParseClientID(text);
                }
                else if (text.find("\"type\":\"broadcast\"") != std::string::npos || text.find("\"type\": \"broadcast\"") != std::string::npos) {
                    // Quick and dirty manual split by player IDs
                    std::vector<std::vector<float>> newlyReceived;
                    size_t playersStart = text.find("\"players\"");
                    if (playersStart != std::string::npos) {
                        // Very naive: find all "transforms": [ ... ] arrays
                        // Since we just extract matrices, we'll scan chunks sequentially 
                        // Wait, we need to skip our own ID.
                        // Let's just find each "{...}" block under players.
                        size_t pos = playersStart;
                        while ((pos = text.find("\"transforms\"", pos)) != std::string::npos) {
                            // Backtrack to find the key which is the client ID
                            size_t keyEnd = text.rfind("\":", pos);
                            size_t keyStart = text.rfind("\"", keyEnd - 1);
                            if (keyStart != std::string::npos && keyEnd != std::string::npos) {
                                int cid = -1;
                                try { cid = std::stoi(text.substr(keyStart+1, keyEnd-keyStart-1)); } catch(...) {}
                                
                                std::vector<float> matrices;
                                if (cid != myClientID && ExtractTransforms(text.substr(pos, 2500), matrices)) {
                                    newlyReceived.push_back(matrices);
                                }
                            }
                            pos += 12; // skip past this transforms instance
                        }
                    }
                    
                    std::lock_guard<std::mutex> lock(stateMutex);
                    latestForeignStates = std::move(newlyReceived);
                }
            } else if (error != ERROR_SUCCESS || bufferType == WINHTTP_WEB_SOCKET_CLOSE_BUFFER_TYPE) {
                break;
            }
        }
    }

public:
    Win32NetworkClient() {}
    
    ~Win32NetworkClient() {
        Disconnect();
    }
    
    void Connect(const std::string& urlString) override {
        Disconnect();
        running = true;
        
        // Parse URL roughly (assuming ws://127.0.0.1:8000/ws)
        // Just hardcoding parsing for local URL or similar format for brevity in this <1MB constraint
        std::wstring wUrl(urlString.begin(), urlString.end());
        URL_COMPONENTS urlComp = {0};
        urlComp.dwStructSize = sizeof(urlComp);
        wchar_t hostName[256]; urlComp.lpszHostName = hostName; urlComp.dwHostNameLength = 256;
        wchar_t urlPath[256]; urlComp.lpszUrlPath = urlPath; urlComp.dwUrlPathLength = 256;
        
        WinHttpCrackUrl(wUrl.c_str(), 0, 0, &urlComp);

        hSession = WinHttpOpen(L"RVGRT WebClient", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        hConnect = WinHttpConnect(hSession, hostName, urlComp.nPort, 0);
        hRequest = WinHttpOpenRequest(hConnect, L"GET", urlPath, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
        
        // Upgrade to websocket
        WinHttpSetOption(hRequest, WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET, NULL, 0);
        WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
        WinHttpReceiveResponse(hRequest, NULL);
        
        hWebSocket = WinHttpWebSocketCompleteUpgrade(hRequest, 0);
        
        hThread = CreateThread(NULL, 0, ReceiveThread, this, 0, NULL);
    }
    
    void Disconnect() override {
        running = false;
        if (hWebSocket) { WinHttpWebSocketClose(hWebSocket, WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS, NULL, 0); WinHttpCloseHandle(hWebSocket); hWebSocket = NULL; }
        if (hRequest) { WinHttpCloseHandle(hRequest); hRequest = NULL; }
        if (hConnect) { WinHttpCloseHandle(hConnect); hConnect = NULL; }
        if (hSession) { WinHttpCloseHandle(hSession); hSession = NULL; }
        if (hThread) { WaitForSingleObject(hThread, 1000); CloseHandle(hThread); hThread = NULL; }
        myClientID = -1;
    }
    
    void SendState(const Character& localCharacter) override {
        if (!hWebSocket || !running) return;
        
        std::stringstream ss;
        ss << "{\"type\":\"state\",\"data\":{\"transforms\":[";
        
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
                ss << val;
                if (p != 6 || i != 15) ss << ",";
            }
        }
        ss << "]}}";
        
        std::string payload = ss.str();
        WinHttpWebSocketSend(hWebSocket, WINHTTP_WEB_SOCKET_UTF8_MESSAGE_BUFFER_TYPE, (PVOID)payload.data(), payload.size());
    }
    
    void PollUpdates(std::vector<Character>& otherCharacters) override {
        std::lock_guard<std::mutex> lock(stateMutex);
        if (latestForeignStates.empty()) return;
        
        otherCharacters.resize(latestForeignStates.size());
        
        for (size_t c = 0; c < latestForeignStates.size(); c++) {
            const std::vector<float>& matrices = latestForeignStates[c];
            float* destPtrs[7] = {
                (float*)&otherCharacters[c].boundingBox.inverseModelMatrix, (float*)&otherCharacters[c].head.inverseModelMatrix,
                (float*)&otherCharacters[c].trunk.inverseModelMatrix, (float*)&otherCharacters[c].leftArm.inverseModelMatrix,
                (float*)&otherCharacters[c].rightArm.inverseModelMatrix, (float*)&otherCharacters[c].leftLeg.inverseModelMatrix,
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
};

std::unique_ptr<NetworkClient> NetworkClient::Create() {
    return std::make_unique<Win32NetworkClient>();
}

#endif // _WIN32
