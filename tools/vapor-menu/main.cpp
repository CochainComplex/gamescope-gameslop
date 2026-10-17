// Optional native Wayland companion; requires the native overlay lifetime fixes.
#include <wayland-client.h>
#include <wayland-egl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <xkbcommon/xkbcommon-keysyms.h>
// Older layer-shell XML uses a C++ keyword as a request argument name.
#define namespace namespace_
#include "layer-shell-client.h"
#undef namespace
#include "action-binding-client.h"
#include "control-client.h"
#include "framegen/hud.hpp"
#include "profiles.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static volatile sig_atomic_t running = 1;
static void stop(int) { running = 0; }
static void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

// CPU artwork is uploaded only on configure or an action, never every vblank.
struct Artwork {
    int width, height;
    std::vector<unsigned char> pixels;
    Artwork(int w, int h) : width(w), height(h), pixels(size_t(w) * h * 4) {}
    void rect(int x, int y, int w, int h, unsigned rgb, unsigned alpha = 255) {
        for (int yy = std::max(0, y); yy < std::min(height, y+h); ++yy)
            for (int xx = std::max(0, x); xx < std::min(width, x+w); ++xx) {
                auto *p = &pixels[(size_t(yy)*width+xx)*4];
                p[0] = ((rgb >> 16) & 255) * alpha / 255;
                p[1] = ((rgb >> 8) & 255) * alpha / 255;
                p[2] = (rgb & 255) * alpha / 255;
                p[3] = alpha;
            }
    }
    void blendRect(int x, int y, int w, int h, unsigned rgb, unsigned alpha) {
        auto linear=[](double c) { c/=255.0; return c<=0.04045 ? c/12.92 : std::pow((c+0.055)/1.055,2.4); };
        auto encode=[](double c) { return unsigned(std::clamp(std::lround(255.0*(c<=0.0031308 ? 12.92*c : 1.055*std::pow(c,1.0/2.4)-0.055)),0l,255l)); };
        for (int yy=std::max(0,y); yy<std::min(height,y+h); ++yy)
            for (int xx=std::max(0,x); xx<std::min(width,x+w); ++xx) {
                auto *p=&pixels[(size_t(yy)*width+xx)*4];
                for (int c=0;c<3;++c) {
                    double base=linear(p[c]), ink=linear((rgb>>(16-c*8))&255);
                    p[c]=encode(base+(ink-base)*alpha/255.0);
                }
                p[3]=255;
            }
    }
    void text(int x, int y, const std::string &s, unsigned color, int scale = 2, unsigned alpha=255) {
        for (unsigned char c : s) {
            auto glyph = gamescope::framegen::k_uFramegenHudFont8x8[c < 128 ? c : '?'];
            for (int row = 0; row < 8; ++row)
                for (int col = 0; col < 8; ++col)
                    if ((glyph >> ((7-row)*8+col)) & 1) {
                        if (alpha==255) rect(x+col*scale, y+row*scale, scale, scale, color);
                        else blendRect(x+col*scale,y+row*scale,scale,scale,color,alpha);
                    }
            x += 8*scale;
        }
    }
};

static constexpr std::array<const char *, 19> labels = {
    "RESUME GAME", "PRESET", "PIPELINE", "MULTIPLIER", "MOTION MODE",
    "FRAME GENERATION", "ONLINE TRAINING", "BIDIRECTIONAL", "LEARNED PROFILE",
    "HUD", "HUD POSITION", "SAVE PRESET", "SAVE AS NEW PRESET", "TAKE SCREENSHOT", "MENU STYLE", "HUD BG OPACITY", "HUD TEXT SIZE", "NPU TRAINING [EXPERIMENTAL]", "GPU DETAILS"
};
static constexpr std::array<const char *,9> settingKeys{
    "pipeline","multiplier","mode","framegen","online","bidir","net_profile","hud","corner"};
static std::string disabledReason(int row, const vapor::Store *store) {
    if (row==18) return "";
    if (row==17) return "NPU training unavailable / offline inference only";
    return store && row>=2 && row<=10
        ? store->draft.disabledReason(settingKeys[row-2],store->context) : "";
}
static std::string rowValue(int row, const vapor::Store *store) {
    if (row==18) return "";
    if (row==17) return "[ ] off / unavailable";
    if (!store) {
        static const std::array<const char *,17> preview{"", "default", "learned", "x2", "motion", "on", "GPU / on", "off", "learned.bin", "compact", "br", "", "", "", "quiet", "auto", "auto"};
        return preview[row];
    }
    if (row==15 || row==16) {
        const auto &value=store->draft.values.at(row==15 ? "hud_opacity" : "hud_scale");
        return value=="auto" ? value : value+(row==15 ? "%" : "x");
    }
    if (row==14) return store->draft.values.at("style");
    if (row==1) return store->name+(store->name=="spatial-experimental" ? " [EXPERIMENTAL]" : "")+(store->dirty ? " *" : "");
    if (row>=2 && row<=10) {
        if (!disabledReason(row,store).empty()) return "inactive";
        auto value=store->draft.values.at(settingKeys[row-2]);
        if (row==2 && value=="spatial") return "spatial [EXPERIMENTAL]";
        if (row==6) return value=="on" ? "GPU / on" : "off";
        if (row==3) return "x"+value;
        if (row==8) return std::filesystem::path(value).filename().string();
        if (row==9) return value=="0" ? "off" : value=="1" ? "compact" : "details";
        return value;
    }
    return "";
}
static Artwork paint(int width, int height, int selected, bool mallsoft, int opacity, const vapor::Store *store=nullptr, bool cga=false, bool experiments=false, bool diagnostics=false, const gamescope::framegen::GpuDiagnostics *gpu=nullptr) {
    Artwork a(width, height);
    if (store) {
        mallsoft=store->draft.values.at("style")=="mallsoft";
        cga=store->draft.values.at("style")=="cga";
    }
    // Fit a complete dashboard, keeping integer pixel-font scaling.
    const int scale=std::clamp(std::min(width/608,height/340),1,4);
    const int w=608*scale, h=340*scale, x=(width-w)/2, y=(height-h)/2;
    const unsigned ink=cga ? 0xffffff : mallsoft ? 0x303c40 : 0xf0eaf5;
    const unsigned secondary=cga ? 0xe0e9ed : mallsoft ? 0x45585b : 0xc1bdce;
    const unsigned accent=cga ? 0x55ffff : mallsoft ? 0x316569 : 0xb3d9da;
    const unsigned rose=cga ? 0xffaaff : mallsoft ? 0x80465f : 0xe3b5ce;
    const unsigned field=cga ? 0x11121c : mallsoft ? 0xdeddd5 : 0x201d2d;
    const unsigned focus=cga ? 0x242538 : mallsoft ? 0xc1cfca : 0x373346;
    auto rect=[&](int px,int py,int ww,int hh,unsigned color,unsigned alpha=255) {
        a.rect(x+px*scale,y+py*scale,ww*scale,hh*scale,color,alpha);
    };
    auto text=[&](int px,int py,const std::string &s,unsigned color) {
        a.text(x+px*scale,y+py*scale,s,color,scale);
    };
    auto shortened=[](std::string value,size_t limit) {
        return value.size()>limit ? value.substr(0,limit-3)+"..." : value;
    };
    rect(0,0,608,340,mallsoft ? 0xd3cfc5 : 0x151127,opacity);
    if (cga) {
        for (int yy=0;yy<340;yy+=2) for (int xx=0;xx<608;xx+=2)
            rect(xx,yy,2,2,gamescope::framegen::framegen_hud_cga_color(xx/2,yy/2,170),opacity);
        // Dither remains in the margins; quiet solid fields support readable text.
        rect(10,8,588,52,field,248);
        rect(10,66,288,184,field,248);
        rect(310,66,288,184,field,248);
        rect(10,256,588,76,field,248);
    }
    rect(0,0,608,1,rose); rect(0,339,608,1,accent);
    if (mallsoft) {
        // Washed glass divisions, without decoration behind the controls.
        for (int i=0;i<4;++i) rect(480+i*30,8,1,16,0x99aaa5,150);
    }
    if (diagnostics) {
        text(18,14,"gamescope-gameslop",accent);
        text(314,14,"GPU DIAGNOSTICS",rose);
        rect(14,32,580,23,field);
        text(22,40,"READ ONLY / LAST COMPOSITED BASE BUFFER",secondary);
        auto lines=[&](int px,int py,std::string value,size_t columns,int count,unsigned color) {
            if (value.empty()) value="unavailable";
            for (int i=0;i<count && !value.empty();++i) {
                const auto part=i==count-1 ? shortened(value,columns) : value.substr(0,columns);
                text(px,py+i*12,part,color);
                value=value.size()>columns ? value.substr(columns) : "";
            }
        };
        auto card=[&](int px,const std::string &heading,uint32_t index) {
            text(px,70,heading,rose);
            if (!gpu || index>=gpu->cards.size()) {
                text(px,92,"Not identified",ink);
                text(px,118,gpu ? "No reliable device address" : "Host diagnostics unavailable",secondary);
                if (gpu) {
                    text(px,152,"VISIBLE GPU CANDIDATES",rose);
                    for (size_t i=0;i<std::min<size_t>(gpu->cards.size(),4);++i)
                        text(px,170+int(i)*16,shortened(gpu->cards[i].name,33),secondary);
                }
                return;
            }
            const auto &info=gpu->cards[index];
            lines(px,90,info.name,33,2,ink);
            text(px,124,"PCI ADDRESS",secondary); text(px,136,info.pci.empty() ? "unavailable / non-PCI" : info.pci,ink);
            char ids[48]; std::snprintf(ids,sizeof(ids),"VENDOR:DEVICE  %04x:%04x",info.vendor,info.device);
            text(px,156,ids,ink);
            text(px,176,"DRM RENDER NODE",secondary); lines(px,188,info.node,33,1,ink);
            text(px,208,"VULKAN DRIVER",secondary); lines(px,220,info.driver,33,2,ink);
        };
        card(22,"PRESENTER / COMPOSITOR",gpu ? gpu->presenter : UINT32_MAX);
        card(322,gpu && gpu->renderer<gpu->cards.size() ? "GAME RENDER / INFERRED" : "GAME RENDER / UNKNOWN",gpu ? gpu->renderer : UINT32_MAX);
        text(18,258,shortened(gpu ? gpu->scope : "This host does not expose GPU diagnostics",71),secondary);
        text(18,274,shortened(gpu ? gpu->evidence : "Use a host built with the optional diagnostics interface",71),rose);
        text(18,290,shortened("BUFFER PATH: "+(gpu ? gpu->transport : "unknown"),71),ink);
        text(18,310,"ENTER REFRESH   ESC SETTINGS   CTRL+SHIFT+F12 CLOSE",accent);
        text(18,325,shortened("BUILD: "+(gpu ? gpu->build : "unknown"),71),secondary);
        return a;
    }
    text(18,14,"gamescope-gameslop",accent);
    text(314,14,"SETTINGS / NEXT LAUNCH",secondary);
    rect(14,32,580,23,selected==1 ? focus : field);
    if (selected==1) rect(14,32,2,23,accent);
    text(22,40,"PRESET",selected==1 ? accent : secondary);
    text(118,40,shortened(experiments ? "spatial-experimental [EXPERIMENTAL]" : rowValue(1,store),58),ink);
    text(22,70,"FRAME GENERATION",rose);
    text(322,70,"HUD / APPEARANCE",rose);
    constexpr std::array<int,8> generation{5,4,2,3,6,17,7,8};
    constexpr std::array<int,5> appearance{9,10,14,15,16};
    auto control=[&](int row,int column,int slot) {
        const int px=column ? 314 : 14, py=86+slot*(column ? 23 : 20);
        const bool active=selected==row;
        bool disabled=row==17;
        disabled=!disabledReason(row,store).empty();
        rect(px,py,280,column ? 21 : 19,active ? focus : field,cga ? 255 : 245);
        if (active) rect(px,py,2,column ? 21 : 19,accent);
        text(px+8,py+2,labels[row],disabled ? secondary : active ? accent : secondary);
        auto value=rowValue(row,store);
        if (experiments && row==2) value="spatial [EXPERIMENTAL]";
        if (!store && row==14) value=cga ? "cga" : mallsoft ? "mallsoft" : "quiet";
        text(px+8,py+12,shortened(value,33),disabled ? secondary : ink);
    };
    for (size_t i=0;i<generation.size();++i) control(generation[i],0,int(i));
    for (size_t i=0;i<appearance.size();++i) control(appearance[i],1,int(i));
    text(322,236,"EXPERIMENTAL = UNVALIDATED",rose);
    constexpr std::array<int,5> actions{0,11,12,13,18};
    constexpr std::array<const char *,5> captions{"RESUME","SAVE PRESET","SAVE AS NEW","SCREENSHOT","GPU DETAILS"};
    for (int i=0;i<5;++i) {
        const int px=14+i*116;
        rect(px,258,112,23,selected==actions[i] ? focus : field);
        if (selected==actions[i]) rect(px,258,2,23,accent);
        text(px+8,266,captions[i],selected==actions[i] ? accent : ink);
    }
    auto status=store ? store->status : "Saved settings apply next launch";
    if (auto reason=disabledReason(selected,store); !reason.empty()) status=reason;
    if (selected==17) status="NPU training unavailable / offline inference only";
    text(18,291,shortened(status,71),rose);
    text(18,310,"UP/DOWN FOCUS  LEFT/RIGHT EDIT  ENTER ACT  TAB GROUP  ESC CLOSE",secondary);
    text(18,325,"CTRL+SHIFT+F12",secondary);
    return a;
}

struct App;
static void configure(void *, zwlr_layer_surface_v1 *, uint32_t, uint32_t, uint32_t);
static void closed(void *, zwlr_layer_surface_v1 *);
struct Action {
    App *app = nullptr;
    int index = 0;
    gamescope_action_binding *binding = nullptr;
    Clock::time_point rearm{};
};
struct App {
    wl_display *display = nullptr;
    wl_compositor *compositor = nullptr;
    zwlr_layer_shell_v1 *shell = nullptr;
    gamescope_action_binding_manager *manager = nullptr;
    gamescope_control *control = nullptr;
    gamescope_gpu_diagnostics_v1 *gpuProtocol = nullptr;
    gamescope::framegen::GpuDiagnostics gpuSnapshot, gpuPending;
    bool diagnosticsOpen = false, gpuReady = false, gpuInFlight = false;
    void refreshGpu() {
        if (!gpuProtocol || gpuInFlight) return;
        gpuInFlight=true;
        if (!gpuReady) {
            gpuSnapshot.scope="Querying current Gamescope GPU information";
            gpuSnapshot.evidence="Waiting for host snapshot";
            gpuReady=true;
        }
        gpuPending={};
        gamescope_gpu_diagnostics_v1_get_snapshot(gpuProtocol);
    }
    wl_surface *surface = nullptr;
    zwlr_layer_surface_v1 *layer = nullptr;
    wl_egl_window *window = nullptr;
    EGLDisplay egl = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface target = EGL_NO_SURFACE;
    EGLConfig config{};
    GLuint program = 0, texture = 0;
    std::array<Action, 8> actions{};
    bool visible = false, configured = false, dirty = false;
    int width = 1280, height = 720, selected = 0, opacity = 235;
    vapor::Store profiles;

    void closeSurface() {
        if (egl != EGL_NO_DISPLAY) {
            eglMakeCurrent(egl, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (target != EGL_NO_SURFACE) eglDestroySurface(egl, target);
            if (context != EGL_NO_CONTEXT) eglDestroyContext(egl, context);
            eglTerminate(egl);
        }
        if (window) wl_egl_window_destroy(window);
        if (layer) zwlr_layer_surface_v1_destroy(layer);
        if (surface) wl_surface_destroy(surface);
        egl=EGL_NO_DISPLAY; context=EGL_NO_CONTEXT; target=EGL_NO_SURFACE;
        window=nullptr; layer=nullptr; surface=nullptr; configured=false;
    }
    ~App() {
        closeSurface();
        for (auto &a : actions) if (a.binding) gamescope_action_binding_destroy(a.binding);
        if (gpuProtocol) gamescope_gpu_diagnostics_v1_destroy(gpuProtocol);
        if (control) gamescope_control_destroy(control);
        if (manager) gamescope_action_binding_manager_destroy(manager);
        if (shell) zwlr_layer_shell_v1_destroy(shell);
        if (compositor) wl_compositor_destroy(compositor);
        if (display) { wl_display_flush(display); wl_display_disconnect(display); }
    }
    void arm(Action &a) {
        // This repository's Execute() currently interprets NO_BLOCK as consume
        // (the inverse of the XML description). One-shot bounds chord repeats.
        gamescope_action_binding_arm(a.binding, GAMESCOPE_ACTION_BINDING_ARM_FLAG_ONE_SHOT |
            GAMESCOPE_ACTION_BINDING_ARM_FLAG_NO_BLOCK);
        a.rearm = {};
    }
    void toggle() {
        visible = !visible;
        if (visible) {
            surface=wl_compositor_create_surface(compositor);
            wl_region *empty=wl_compositor_create_region(compositor);
            wl_surface_set_input_region(surface,empty); wl_region_destroy(empty);
            layer=zwlr_layer_shell_v1_get_layer_surface(shell,surface,nullptr,
                ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY,"gamescope-vapor-menu");
            zwlr_layer_surface_v1_set_anchor(layer,15);
            zwlr_layer_surface_v1_set_size(layer,0,0);
            zwlr_layer_surface_v1_set_exclusive_zone(layer,-1);
            zwlr_layer_surface_v1_set_keyboard_interactivity(layer,0);
            static const zwlr_layer_surface_v1_listener layerListener{configure,closed};
            zwlr_layer_surface_v1_add_listener(layer,&layerListener,this);
            wl_surface_commit(surface);
            if (diagnosticsOpen) refreshGpu();
        } else closeSurface();
        for (size_t i=1; i<actions.size(); ++i) {
            actions[i].rearm = {};
            if (visible) arm(actions[i]);
            else gamescope_action_binding_disarm(actions[i].binding);
        }
        dirty = true;
    }
    void action(int index) {
        try {
        if (index == 0) { toggle(); return; }
        if (!visible) return;
        if (diagnosticsOpen) {
            if (index==1) { diagnosticsOpen=false; dirty=true; }
            if (index==6) refreshGpu();
            return;
        }
        if (index == 1) { toggle(); return; }
        // Visual order: preset, generation column, appearance column, actions.
        static constexpr std::array<int,19> order{1,5,4,2,3,6,17,7,8,9,10,14,15,16,0,11,12,13,18};
        if (index==2 || index==3) {
            auto position=std::find(order.begin(),order.end(),selected)-order.begin();
            selected=order[(position+(index==2 ? int(order.size())-1 : 1))%order.size()];
        }
        if (index==7) {
            if (selected==1) selected=5;
            else if ((selected>=2 && selected<=8) || selected==17) selected=9;
            else if (selected==9 || selected==10 || (selected>=14 && selected<=16)) selected=0;
            else selected=1;
        }
        if (index == 4 || index == 5 || index == 6) {
            int delta = index == 4 ? -1 : 1;
            if (selected==18) {
                if (index==6) { diagnosticsOpen=true; refreshGpu(); dirty=true; }
                return;
            }
            if (selected==17) { profiles.status="NPU training unavailable / offline inference only"; dirty=true; return; }
            if (selected == 1) profiles.select(delta);
            if (selected>=2 && selected<=10 && selected!=8) {
                profiles.edit(settingKeys[selected-2],delta);
            }
            if (selected==14) {
                profiles.draft.cycle("style",delta); profiles.dirty=true;
                profiles.status="Menu preview / save; HUD on next launch";
            }
            if (selected==8) profiles.cycleLearned(delta);
            if (index==6 && selected==11) profiles.persist();
            if (index==6 && selected==12) profiles.clone();
            if (index == 6 && selected == 0) toggle();
            if (index == 6 && selected == 13) {
                std::string path = "/tmp/gamescope-menu-" + std::to_string(
                    std::chrono::system_clock::now().time_since_epoch().count()) + ".png";
                // Base-plane capture deliberately excludes this menu.
                gamescope_control_take_screenshot(control, path.c_str(),
                    GAMESCOPE_CONTROL_SCREENSHOT_TYPE_BASE_PLANE_ONLY, 0);
                std::fprintf(stderr, "Screenshot requested: %s\n", path.c_str());
                toggle();
            }
        }
        } catch (const std::exception &e) { profiles.status=e.what(); }
        dirty = true;
    }
    static void triggered(void *data, gamescope_action_binding *, uint32_t, uint32_t, uint32_t, uint32_t) {
        auto &a = *static_cast<Action *>(data);
        a.app->action(a.index);
        if (a.index == 0 || a.app->visible)
            a.rearm = Clock::now()+std::chrono::milliseconds(250);
    }
    void bindings() {
        static const gamescope_action_binding_listener listener{triggered};
        const std::array<uint32_t, 7> keys{XKB_KEY_Escape, XKB_KEY_Up, XKB_KEY_Down,
            XKB_KEY_Left, XKB_KEY_Right, XKB_KEY_Return, XKB_KEY_Tab};
        for (size_t i=0; i<actions.size(); ++i) {
            auto &a = actions[i]; a.app=this; a.index=int(i);
            a.binding = gamescope_action_binding_manager_create_action_binding(manager);
            gamescope_action_binding_add_listener(a.binding, &listener, &a);
            gamescope_action_binding_set_description(a.binding, i == 0 ? "Vapor menu toggle" : "Vapor menu navigation");
            auto add = [&](const std::vector<uint32_t> &syms) {
                wl_array arr{syms.size()*sizeof(uint32_t), 0, const_cast<uint32_t *>(syms.data())};
                gamescope_action_binding_add_keyboard_trigger(a.binding, &arr);
            };
            if (i == 0) {
                for (auto ctrl : {XKB_KEY_Control_L, XKB_KEY_Control_R})
                    for (auto shift : {XKB_KEY_Shift_L, XKB_KEY_Shift_R})
                        add({uint32_t(ctrl), uint32_t(shift), XKB_KEY_F12});
                arm(a);
            } else add({keys[i-1]});
        }
    }
    void initEgl() {
        egl = eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, display, nullptr);
        require(egl != EGL_NO_DISPLAY && eglInitialize(egl, nullptr, nullptr), "Cannot initialize Wayland EGL");
        require(eglBindAPI(EGL_OPENGL_ES_API), "Cannot bind OpenGL ES");
        const EGLint attrs[]{EGL_SURFACE_TYPE,EGL_WINDOW_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,
            EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
        EGLint count=0;
        require(eglChooseConfig(egl,attrs,&config,1,&count) && count, "No RGBA EGL config");
        const EGLint ctx[]{EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE};
        context=eglCreateContext(egl,config,EGL_NO_CONTEXT,ctx);
        require(context != EGL_NO_CONTEXT, "Cannot create EGL context");
        window=wl_egl_window_create(surface,width,height);
        require(window, "Cannot create native Wayland EGL window");
        target=eglCreateWindowSurface(egl,config,window,nullptr);
        require(target != EGL_NO_SURFACE && eglMakeCurrent(egl,target,target,context), "Cannot create EGL surface");
        auto shader = [](GLenum type, const char *source) {
            GLuint s=glCreateShader(type); glShaderSource(s,1,&source,nullptr); glCompileShader(s);
            GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
            require(ok, "Menu shader compilation failed"); return s;
        };
        GLuint vs=shader(GL_VERTEX_SHADER,"attribute vec2 p; varying vec2 uv; void main(){gl_Position=vec4(p,0.,1.);uv=vec2((p.x+1.)*.5,(1.-p.y)*.5);}");
        GLuint fs=shader(GL_FRAGMENT_SHADER,"precision mediump float; varying vec2 uv; uniform sampler2D image; void main(){gl_FragColor=texture2D(image,uv);}");
        program=glCreateProgram(); glAttachShader(program,vs); glAttachShader(program,fs);
        glBindAttribLocation(program,0,"p"); glLinkProgram(program);
        GLint ok=0; glGetProgramiv(program,GL_LINK_STATUS,&ok); require(ok,"Menu shader link failed");
        glDeleteShader(vs); glDeleteShader(fs); glUseProgram(program);
        glGenTextures(1,&texture); glBindTexture(GL_TEXTURE_2D,texture);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    }
    void draw() {
        if (!configured || !dirty) return;
        if (!window) initEgl();
        wl_egl_window_resize(window,width,height,0,0);
        glViewport(0,0,width,height);
        Artwork image = visible ? paint(width,height,selected,false,opacity,&profiles,false,false,diagnosticsOpen,gpuReady ? &gpuSnapshot : nullptr) : Artwork(width,height);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,image.pixels.data());
        const GLfloat vertices[]{-1,-1, 1,-1, -1,1, 1,1};
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,vertices); glEnableVertexAttribArray(0);
        glDrawArrays(GL_TRIANGLE_STRIP,0,4);
        require(glGetError()==GL_NO_ERROR && eglSwapBuffers(egl,target), "Menu rendering failed");
        dirty=false;
    }
};

static void global(void *data, wl_registry *r, uint32_t id, const char *name, uint32_t version) {
    auto &a=*static_cast<App *>(data);
    if (!std::strcmp(name,"wl_compositor")) a.compositor=static_cast<wl_compositor *>(wl_registry_bind(r,id,&wl_compositor_interface,std::min(version,3u)));
    if (!std::strcmp(name,"zwlr_layer_shell_v1")) a.shell=static_cast<zwlr_layer_shell_v1 *>(wl_registry_bind(r,id,&zwlr_layer_shell_v1_interface,1));
    if (!std::strcmp(name,"gamescope_action_binding_manager")) a.manager=static_cast<gamescope_action_binding_manager *>(wl_registry_bind(r,id,&gamescope_action_binding_manager_interface,1));
    if (!std::strcmp(name,"gamescope_gpu_diagnostics_v1")) {
        a.gpuProtocol=static_cast<gamescope_gpu_diagnostics_v1 *>(wl_registry_bind(r,id,&gamescope_gpu_diagnostics_v1_interface,1));
        static const gamescope_gpu_diagnostics_v1_listener listener{
            [](void *data,gamescope_gpu_diagnostics_v1 *,uint32_t index,uint32_t vendor,uint32_t device,uint32_t software,
               const char *name,const char *pci,const char *node,const char *driver) {
                auto &app=*static_cast<App *>(data);
                if (index!=app.gpuPending.cards.size() || index>=16) return;
                app.gpuPending.cards.push_back({name,pci,node,driver,vendor,device,software!=0});
            },
            [](void *data,gamescope_gpu_diagnostics_v1 *,uint32_t present,uint32_t render,const char *evidence,
               const char *transport,const char *scope,const char *build) {
                auto &value=static_cast<App *>(data)->gpuPending;
                value.presenter=present; value.renderer=render; value.evidence=evidence;
                value.transport=transport; value.scope=scope; value.build=build;
            },
            [](void *data,gamescope_gpu_diagnostics_v1 *) {
                auto &app=*static_cast<App *>(data); app.gpuSnapshot=std::move(app.gpuPending);
                app.gpuReady=true; app.gpuInFlight=false; app.dirty=true;
            },
        };
        gamescope_gpu_diagnostics_v1_add_listener(a.gpuProtocol,&listener,&a);
    }
    if (!std::strcmp(name,"gamescope_control") && version>=3) {
        a.control=static_cast<gamescope_control *>(wl_registry_bind(r,id,&gamescope_control_interface,3));
        static const gamescope_control_listener controlListener{
            [](void *, gamescope_control *, uint32_t, uint32_t, uint32_t) {},
            [](void *, gamescope_control *, const char *, const char *, const char *, uint32_t, wl_array *) {},
            [](void *, gamescope_control *, const char *path) { std::fprintf(stderr,"Screenshot saved: %s\n",path); },
            nullptr,
        };
        gamescope_control_add_listener(a.control,&controlListener,&a);
    }
}
static void removed(void *, wl_registry *, uint32_t) {}
static void configure(void *data, zwlr_layer_surface_v1 *layer, uint32_t serial, uint32_t w, uint32_t h) {
    auto &a=*static_cast<App *>(data);
    zwlr_layer_surface_v1_ack_configure(layer,serial);
    if (w && h && w<=16384 && h<=16384) { a.width=int(w); a.height=int(h); }
    a.configured=true; a.dirty=true;
}
static void closed(void *, zwlr_layer_surface_v1 *) { running=0; }

static gamescope::framegen::GpuDiagnostics preview_gpu_diagnostics() {
    gamescope::framegen::GpuDiagnostics value;
    value.cards={{"AMD Radeon 890M Graphics","0000:66:00.0","/dev/dri/renderD128","RADV / Mesa 25.2",0x1002,0x150e,false},
        {"NVIDIA RTX PRO 500 Laptop GPU","0000:01:00.0","/dev/dri/renderD129","NVIDIA / proprietary Vulkan",0x10de,0x2db9,false}};
    value.presenter=0; value.scope="Nested swapchain: outer display GPU unknown"; value.build="preview+96ad25e89";
    return gamescope::framegen::resolve_gpu_diagnostics(std::move(value),gamescope::framegen::gpu_buffer_observation("NVIDIA",true));
}

int main(int argc, char **argv) {
    try {
        if (argc==3 && (!std::strcmp(argv[1],"--preview-gpu-details-ppm") || !std::strcmp(argv[1],"--preview-gpu-details-cga-ppm") || !std::strcmp(argv[1],"--preview-gpu-details-mallsoft-ppm") || !std::strcmp(argv[1],"--preview-gpu-unknown-ppm") || !std::strcmp(argv[1],"--preview-experimental-ppm") || !std::strcmp(argv[1],"--preview-ppm") || !std::strcmp(argv[1],"--preview-hud-ppm") || !std::strcmp(argv[1],"--preview-mallsoft-ppm") || !std::strcmp(argv[1],"--preview-hud-details-ppm") || !std::strcmp(argv[1],"--preview-hud-off-ppm") || !std::strcmp(argv[1],"--preview-cga-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-details-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-clear-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-opaque-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-large-ppm"))) {
            auto image=paint(1280,720,2,!std::strcmp(argv[1],"--preview-mallsoft-ppm"),235,nullptr,!std::strcmp(argv[1],"--preview-cga-ppm") || !std::strcmp(argv[1],"--preview-experimental-ppm"),!std::strcmp(argv[1],"--preview-experimental-ppm"));
            if (std::strstr(argv[1],"--preview-gpu-")==argv[1]) {
                auto gpu=preview_gpu_diagnostics();
                if (!std::strcmp(argv[1],"--preview-gpu-unknown-ppm"))
                    gpu=gamescope::framegen::resolve_gpu_diagnostics(std::move(gpu),gamescope::framegen::gpu_buffer_observation("linear",false));
                image=paint(1280,720,18,!std::strcmp(argv[1],"--preview-gpu-details-mallsoft-ppm"),235,nullptr,
                    !std::strcmp(argv[1],"--preview-gpu-details-cga-ppm"),false,true,&gpu);
            }
            if (!std::strcmp(argv[1],"--preview-hud-ppm") || !std::strcmp(argv[1],"--preview-hud-details-ppm") || !std::strcmp(argv[1],"--preview-hud-off-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-details-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-clear-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-opaque-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-large-ppm")) {
                // Layout preview uses the runtime formatter/font. GPU color
                // management is verified separately; this is not a game capture.
                image=Artwork(1280,720);
                image.rect(0,0,1280,720,0x1c2431);
                // Muted bands make the backing's transparency visible.
                for (int row=0;row<720;++row)
                    image.rect(0,row,1280,1,((28+row/30)<<16)|((36+row/24)<<8)|(49+row/18));
                for (int i=0;i<8;++i) image.rect(i*180,610,90,110,0x374054);
                const gamescope::framegen::FramegenHudSnapshot_t snapshot{
                    .version="preview+96ad25e89", .deviceName="AMD Radeon Graphics",
                    .gpu=preview_gpu_diagnostics(),
                    .mode=GamescopeFramegenMode::Motion,
                    .pipeline=GamescopeFramegenPipeline::Learned,
                    .multiplier=2u, .refreshMilliHz=120000u,
                    .clientBuffersStaged=true, .netActive=true, .netOnline=true,
                    .real=200u, .generated=390u, .repeats=10u,
                    .deadlineHitPercent=99u, .pacingSdTenthsMs=7u,
                    .deadlineHitHistory={95,97,99,100,99,98,99,100,100,99,99,100},
                    .pacingSdMsHistory={0.4,0.7,1.1,0.9,0.5,0.6,0.7,0.5,0.8,0.7,0.6,0.7},
                    .netTrainedSteps=1200u, .netProfileLoaded=true,
                };
                auto text=gamescope::framegen::format_framegen_hud(
                    !std::strcmp(argv[1],"--preview-hud-off-ppm") ? 0u :
                    (!std::strcmp(argv[1],"--preview-hud-details-ppm") || !std::strcmp(argv[1],"--preview-hud-cga-details-ppm")) ? 2u : 1u,snapshot);
                if (text.lineCount>0) {
                const bool cgaHud=std::strstr(argv[1],"--preview-hud-cga")!=nullptr;
                const int fontScale=!std::strcmp(argv[1],"--preview-hud-cga-large-ppm") ? 3 : 2;
                const uint32_t opacity=!std::strcmp(argv[1],"--preview-hud-cga-clear-ppm") ? 0u :
                    !std::strcmp(argv[1],"--preview-hud-cga-opaque-ppm") ? 100u : 101u;
                auto uniform=gamescope::framegen::make_framegen_hud_uniform(text,false,cgaHud,opacity);
                const int pitch=int(gamescope::framegen::framegen_hud_line_pitch(cgaHud));
                const int w=(int(uniform.widthChars)*8+8)*fontScale;
                const int h=int(gamescope::framegen::framegen_hud_height(text.lineCount,cgaHud))*fontScale;
                const int x=image.width-w-8, y=image.height-h-8;
                const unsigned alpha=opacity<=100u ? (opacity*255u+50u)/100u : cgaHud ? 92u : 36u;
                if (cgaHud) {
                    for (int yy=0;yy<h;yy+=fontScale) for (int xx=0;xx<w;xx+=fontScale)
                        image.blendRect(x+xx,y+yy,fontScale,fontScale,
                            gamescope::framegen::framegen_hud_cga_color(xx/fontScale,yy/fontScale,h/fontScale),alpha);
                    for (uint32_t i=0;i<text.lineCount;++i)
                        image.blendRect(x+3*fontScale,y+(3+int(i)*pitch)*fontScale,
                            (int(uniform.lineLengths[i])*8+2)*fontScale,10*fontScale,0x000000,alpha*65u/100u);
                } else image.blendRect(x,y,w,h,0x110d1b,alpha);
                image.blendRect(x,y,w,fontScale,cgaHud ? 0xffffff : 0x80e0e6,122);
                image.blendRect(x,y+fontScale,fontScale,h-fontScale,cgaHud ? 0xffffff : 0xe066b8,97);
                for (uint32_t i=0;i<text.lineCount;++i) {
                    const int tx=x+4*fontScale, ty=y+(4+int(i)*pitch)*fontScale;
                    if (cgaHud) {
                        image.text(tx-fontScale,ty,text.lines[i].data(),0x000000,fontScale,250);
                        image.text(tx+fontScale,ty,text.lines[i].data(),0x000000,fontScale,250);
                        image.text(tx,ty-fontScale,text.lines[i].data(),0x000000,fontScale,250);
                        image.text(tx,ty+fontScale,text.lines[i].data(),0x000000,fontScale,250);
                    }
                    for (uint32_t col=0;col<uniform.lineLengths[i];++col) {
                        bool value=gamescope::framegen::framegen_hud_value_column(text.lines[i].data(),col);
                        image.text(tx+int(col)*8*fontScale,ty,std::string(1,text.lines[i][col]),
                            value ? (cgaHud ? 0xffffff : 0xf5f0fa) : (cgaHud ? 0x55ffff : 0xa1ccd1),
                            fontScale,value || cgaHud ? 250 : 230);
                    }
                }
                }
            }
            std::ofstream out(argv[2],std::ios::binary); out << "P6\n1280 720\n255\n";
            for (size_t i=0;i<image.pixels.size();i+=4) out.write(reinterpret_cast<const char *>(&image.pixels[i]),3);
            require(bool(out),"Cannot write preview"); return 0;
        }
        bool smoke=argc==2 && !std::strcmp(argv[1],"--smoke");
        if (argc!=1 && !smoke) { std::puts("Usage: vapor-menu [--preview-ppm PATH | --preview-mallsoft-ppm PATH | --preview-hud-ppm PATH | --preview-hud-details-ppm PATH | --preview-hud-off-ppm PATH | --smoke]\nCtrl+Shift+F12 toggle; arrows navigate; Enter apply; Escape resume."); return argc==2 && !std::strcmp(argv[1],"--help") ? 0 : 1; }
        App a;
        a.display=wl_display_connect(nullptr); require(a.display,"Cannot connect to Wayland; launch INSIDE gamescope");
        wl_registry *registry=wl_display_get_registry(a.display);
        static const wl_registry_listener listener{global,removed};
        wl_registry_add_listener(registry,&listener,&a);
        require(wl_display_roundtrip(a.display)>=0,"Wayland registry failed");
        wl_registry_destroy(registry);
        require(a.compositor && a.shell && a.manager && a.control,
            "Requires gamescope Wayland layer-shell, action-binding and control v3; no X11 fallback");
        a.bindings();
        const auto smokeStart=Clock::now();
        int smokeStep=0;
        if (smoke) a.toggle();
        std::signal(SIGINT,stop); std::signal(SIGTERM,stop);
        std::puts("gamescope-gameslop menu ready: Ctrl+Shift+F12 (inside gamescope)");
        while (running) {
            require(wl_display_dispatch_pending(a.display)>=0,"Wayland disconnected");
            int timeout=-1;
            const auto now=Clock::now();
            if (smoke) {
                timeout=50;
                const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(now-smokeStart).count();
                if (ms>4000) { require(smokeStep==4,"Smoke did not finish rendering"); break; }
                if (ms>500 && smokeStep==0 && a.configured && !a.dirty) {
                    // Follow visual order and exercise fast group navigation.
                    a.selected=1; a.action(7); require(a.selected==5,"Generation group navigation failed");
                    a.action(7); require(a.selected==9,"HUD group navigation failed");
                    a.action(7); require(a.selected==0,"Action group navigation failed");
                    a.action(7); require(a.selected==1,"Preset group navigation failed");
                    a.action(3); a.action(3); a.action(3); // Pipeline, after enable/mode.
                    require(a.selected==2,"Visual focus order failed");
                    a.action(5); a.action(3); a.action(5);
                    // Exercise the integrated mode through the same menu actions.
                    a.selected=2;
                    for (size_t i=0; i<vapor::choices.at("pipeline").size() &&
                        a.profiles.draft.values.at("pipeline")!="spatial"; ++i) a.action(5);
                    require(a.profiles.draft.values.at("pipeline")=="spatial" &&
                        a.profiles.draft.values.at("multiplier")=="3" && a.profiles.dirty,
                        "Menu profile edit failed");
                    a.selected=7; a.action(5);
                    require(a.profiles.draft.values.at("bidir")=="off","Spatial allowed bidir");
                    a.selected=4; a.action(5); // Motion -> extrapolate.
                    a.selected=2; a.action(5); // Inactive pipeline must not change.
                    require(a.profiles.draft.values.at("pipeline")=="spatial" &&
                        !disabledReason(2,&a.profiles).empty(),"Inactive pipeline was editable");
                    const auto online=a.profiles.draft.values.at("online");
                    a.selected=6; a.action(5);
                    require(a.profiles.draft.values.at("online")==online,"Inactive training was editable");
                    a.selected=4; a.action(4); // Restore motion.
                    std::puts("Smoke: configuration dependencies PASS");
                    a.selected=18; a.action(6);
                    require(a.diagnosticsOpen,"GPU diagnostics did not open");
                    smokeStep=1;
                }
                if (ms>1000 && smokeStep==1) {
                    require(a.gpuReady && !a.gpuInFlight && !a.gpuSnapshot.cards.empty(),"GPU diagnostics IPC did not finish");
                    require(a.gpuSnapshot.presenter<a.gpuSnapshot.cards.size(),"Missing compositor GPU identity");
                    a.action(1); require(!a.diagnosticsOpen && a.visible,"Diagnostics back navigation failed");
                    std::puts("Smoke: GPU diagnostics request/response/back PASS");
                    a.toggle(); smokeStep=2;
                }
                if (ms>1500 && smokeStep==2) { a.toggle(); smokeStep=3; }
                if (ms>2000 && smokeStep==3 && a.configured && !a.dirty) {
                    require(a.profiles.draft.values.at("multiplier")=="3" &&
                        a.profiles.draft.values.at("pipeline")=="spatial" && a.profiles.dirty,
                        "Menu lost the unsaved draft");
                    a.toggle(); smokeStep=4; std::puts("Smoke: configure/render/navigate/hide/reopen PASS");
                }
            }
            for (auto &action : a.actions) if (action.rearm!=Clock::time_point{}) {
                if (now>=action.rearm) a.arm(action);
                else {
                    int ms=int(std::chrono::duration_cast<std::chrono::milliseconds>(action.rearm-now).count())+1;
                    timeout=timeout<0 ? ms : std::min(timeout,ms);
                }
            }
            a.draw();
            // EGL may have dispatched protocol events while swapping.
            if (wl_display_prepare_read(a.display)!=0) continue;
            wl_display_flush(a.display);
            pollfd fd{wl_display_get_fd(a.display),POLLIN,0};
            int result=poll(&fd,1,timeout);
            if (result>0 && (fd.revents&POLLIN)) require(wl_display_read_events(a.display)>=0,"Wayland read failed");
            else { wl_display_cancel_read(a.display); require(result<=0 || !(fd.revents&(POLLERR|POLLHUP|POLLNVAL)),"Wayland connection closed"); }
        }
    } catch (const std::exception &e) { std::fprintf(stderr,"gamescope-gameslop menu: %s\n",e.what()); return 1; }
    return 0;
}
