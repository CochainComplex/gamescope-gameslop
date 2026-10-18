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

static constexpr std::array<const char *, 14> labels = {
    "RESUME GAME", "PRESET", "PIPELINE", "MULTIPLIER", "MOTION MODE",
    "FRAME GENERATION", "ONLINE TRAINING", "BIDIRECTIONAL", "LEARNED PROFILE",
    "HUD", "HUD POSITION", "SAVE PRESET", "SAVE AS NEW PRESET", "TAKE SCREENSHOT"
};
static std::string rowValue(int row, const vapor::Store *store) {
    if (!store) {
        static const std::array<const char *,14> preview{"", "default", "learned", "x2", "motion", "on", "on", "off", "learned.bin", "compact", "br", "", "", ""};
        return preview[row];
    }
    if (row==1) return store->name+(store->dirty ? " *" : "");
    static const std::array<const char *,9> keys{"pipeline","multiplier","mode","framegen","online","bidir","net_profile","hud","corner"};
    if (row>=2 && row<=10) {
        auto value=store->draft.values.at(keys[row-2]);
        if (row==3) return "x"+value;
        if (row==8) return std::filesystem::path(value).filename().string();
        if (row==9) return value=="0" ? "off" : value=="1" ? "compact" : "details";
        return value;
    }
    return "";
}
static Artwork paint(int width, int height, int selected, bool alternate, int opacity, const vapor::Store *store=nullptr) {
    Artwork a(width, height);
    const unsigned pink = alternate ? 0xffad68 : 0xff65cc;
    const unsigned cyan = alternate ? 0xb5fc8d : 0x7ef9ff;
    const int scale = std::clamp(std::min(width/720, height/480), 1, 3);
    const int w = std::min(width, 448*scale), h = std::min(height, 360*scale);
    const int x = (width-w)/2, y = (height-h)/2;
    a.rect(x+5, y+5, w, h, 0x410b58, 120);
    a.rect(x, y, w, h, 0x151127, opacity);
    a.rect(x, y, w, 2, pink); a.rect(x, y+h-2, w, 2, cyan);
    a.rect(x, y, 2, h, pink); a.rect(x+w-2, y, 2, h, cyan);
    auto text = [&](int px, int py, const std::string &s, unsigned c, int z=1) {
        a.text(x+px*scale, y+py*scale, s, c, scale*z);
    };
    // Striped sunset and a perspective grid, kept away from the menu text.
    for (int row = -22*scale; row <= 22*scale; ++row) {
        int half = int(std::sqrt(double(22*scale*22*scale-row*row)));
        if (row < 0 || (row/scale)%5 < 3)
            a.rect(x+w-52*scale-half, y+49*scale+row, half*2, 1, pink, 210);
    }
    for (int i=0; i<4; ++i)
        a.rect(x+18*scale, y+(286+i*i)*scale, w-36*scale, 1, 0x56366e, 100);
    text(18, 15, "G A M E S C O P E", cyan);
    text(18, 35, "AFTER HOURS", pink, 2);
    text(18, 64, "PROFILE SETTINGS / NEXT LAUNCH", 0xb3a6d1);
    constexpr int page=8;
    const int first=std::clamp(selected-page+1,0,int(labels.size())-page);
    for (int i=first; i<std::min(first+page,int(labels.size())); ++i) {
        const int yy = 91+(i-first)*25;
        if (selected == i) {
            a.rect(x+14*scale, y+(yy-5)*scale, w-28*scale, 21*scale, 0x352044, 210);
            a.rect(x+14*scale, y+(yy-5)*scale, 2*scale, 21*scale, cyan);
        }
        text(24, yy, (selected == i ? "> " : "  ") + std::string(labels[i]), selected == i ? cyan : 0xd6cce9);
        auto value=rowValue(i,store);
        if (value.size()>20) value=value.substr(0,17)+"...";
        text(264, yy, value, pink);
    }
    auto status=store ? store->status : "Saved settings apply next launch";
    if (status.size()>50) status=status.substr(0,47)+"...";
    text(18, 299, status, 0xb3a6d1);
    text(18, 320, "ARROWS EDIT   ENTER SELECT   ESC CLOSE", cyan);
    text(18, 341, "CTRL+SHIFT+F12   "+std::to_string(selected+1)+"/"+std::to_string(labels.size()), 0xa695c2);
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
    wl_surface *surface = nullptr;
    zwlr_layer_surface_v1 *layer = nullptr;
    wl_egl_window *window = nullptr;
    EGLDisplay egl = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface target = EGL_NO_SURFACE;
    EGLConfig config{};
    GLuint program = 0, texture = 0;
    std::array<Action, 7> actions{};
    bool visible = false, configured = false, dirty = false, alternate = false;
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
        if (index == 1) { toggle(); return; }
        if (index == 2) selected = (selected+int(labels.size())-1)%int(labels.size());
        if (index == 3) selected = (selected+1)%int(labels.size());
        if (index == 4 || index == 5 || index == 6) {
            int delta = index == 4 ? -1 : 1;
            if (selected == 1) profiles.select(delta);
            if (selected>=2 && selected<=10 && selected!=8) {
                static const std::array<const char *,9> keys{"pipeline","multiplier","mode","framegen","online","bidir","net_profile","hud","corner"};
                profiles.draft.cycle(keys[selected-2],delta); profiles.dirty=true;
                profiles.status="Unsaved / changes apply next launch";
            }
            if (selected==8) { profiles.cycleLearned(delta); profiles.status="Unsaved / learned file unchanged"; }
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
        const std::array<uint32_t, 6> keys{XKB_KEY_Escape, XKB_KEY_Up, XKB_KEY_Down,
            XKB_KEY_Left, XKB_KEY_Right, XKB_KEY_Return};
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
        Artwork image = visible ? paint(width,height,selected,alternate,opacity,&profiles) : Artwork(width,height);
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

int main(int argc, char **argv) {
    try {
        if (argc==3 && (!std::strcmp(argv[1],"--preview-ppm") || !std::strcmp(argv[1],"--preview-hud-ppm"))) {
            auto image=paint(1280,720,2,false,235);
            if (!std::strcmp(argv[1],"--preview-hud-ppm")) {
                // Layout preview uses the runtime formatter/font. GPU color
                // management is verified separately; this is not a game capture.
                image=Artwork(1280,720);
                image.rect(0,0,1280,720,0x1c2431);
                // Muted bands make the backing's transparency visible.
                for (int row=0;row<720;++row)
                    image.rect(0,row,1280,1,((28+row/30)<<16)|((36+row/24)<<8)|(49+row/18));
                for (int i=0;i<8;++i) image.rect(i*180,610,90,110,0x374054);
                const gamescope::framegen::FramegenHudSnapshot_t snapshot{
                    .mode=GamescopeFramegenMode::Motion,
                    .pipeline=GamescopeFramegenPipeline::Learned,
                    .multiplier=2u, .refreshMilliHz=120000u,
                    .netActive=true, .netOnline=true,
                    .real=200u, .generated=390u, .repeats=10u,
                    .deadlineHitPercent=99u, .pacingSdTenthsMs=7u,
                    .netTrainedSteps=1200u, .netProfileLoaded=true,
                };
                auto text=gamescope::framegen::format_framegen_hud(1u,snapshot);
                auto uniform=gamescope::framegen::make_framegen_hud_uniform(text,false);
                const int w=int(uniform.widthChars)*16+16, h=int(text.lineCount)*16+16;
                const int x=image.width-w-8, y=image.height-h-8;
                image.blendRect(x,y,w,h,0x110d1b,36);
                image.blendRect(x,y,w,2,0x80e0e6,122); image.blendRect(x,y+2,2,h-2,0xe066b8,97);
                for (uint32_t i=0;i<text.lineCount;++i)
                    image.text(x+8,y+8+int(i)*16,text.lines[i].data(),i==0 ? 0xa1dbe0 : 0xbfb5cf,2,219);
            }
            std::ofstream out(argv[2],std::ios::binary); out << "P6\n1280 720\n255\n";
            for (size_t i=0;i<image.pixels.size();i+=4) out.write(reinterpret_cast<const char *>(&image.pixels[i]),3);
            require(bool(out),"Cannot write preview"); return 0;
        }
        bool smoke=argc==2 && !std::strcmp(argv[1],"--smoke");
        if (argc!=1 && !smoke) { std::puts("Usage: vapor-menu [--preview-ppm PATH | --smoke]\nCtrl+Shift+F12 toggle; arrows navigate; Enter apply; Escape resume."); return argc==2 && !std::strcmp(argv[1],"--help") ? 0 : 1; }
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
        std::puts("Vapor menu ready: Ctrl+Shift+F12 (inside gamescope)");
        while (running) {
            require(wl_display_dispatch_pending(a.display)>=0,"Wayland disconnected");
            int timeout=-1;
            const auto now=Clock::now();
            if (smoke) {
                timeout=50;
                const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(now-smokeStart).count();
                if (ms>4000) { require(smokeStep==4,"Smoke did not finish rendering"); break; }
                if (ms>500 && smokeStep==0 && a.configured && !a.dirty) {
                    a.action(3); a.action(3); a.action(5); a.action(3); a.action(5);
                    // Exercise the integrated mode through the same menu actions.
                    a.selected=2;
                    for (size_t i=0; i<vapor::choices.at("pipeline").size() &&
                        a.profiles.draft.values.at("pipeline")!="spatial"; ++i) a.action(5);
                    require(a.profiles.draft.values.at("pipeline")=="spatial" &&
                        a.profiles.draft.values.at("multiplier")=="3" && a.profiles.dirty,
                        "Menu profile edit failed");
                    smokeStep=1;
                }
                if (ms>1000 && smokeStep==1) { a.toggle(); smokeStep=2; }
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
    } catch (const std::exception &e) { std::fprintf(stderr,"vapor-menu: %s\n",e.what()); return 1; }
    return 0;
}
