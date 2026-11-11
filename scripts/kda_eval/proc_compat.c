/* Minimal procfs compatibility for the namespace-free chroot sandbox.
 * The trusted broker serves only maps/status/stat/exe/cwd for sandbox UIDs.
 * Isolation does not depend on this shim; bypassing it never exposes host paths.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

static int proc_path(const char *path) {
    if (!path || strncmp(path,"/proc/",6)) return 0;
    const char *tail=strchr(path+6,'/');
    if (!tail) return 0;
    return !strcmp(tail,"/maps") || !strcmp(tail,"/status") || !strcmp(tail,"/stat")
        || !strcmp(tail,"/exe") || !strcmp(tail,"/cwd");
}

static int query(const char *path,char *result,size_t size) {
    int sock=socket(AF_UNIX,SOCK_SEQPACKET|SOCK_CLOEXEC,0);
    if(sock<0)return -1;
    struct sockaddr_un addr={.sun_family=AF_UNIX};
    strcpy(addr.sun_path,"/run/kh/proc.sock");
    if(connect(sock,(void*)&addr,sizeof(addr)) || send(sock,path,strlen(path),0)<0) {
        close(sock);return -1;
    }
    struct iovec io={.iov_base=result,.iov_len=size};
    char control[CMSG_SPACE(sizeof(int))]={0};
    struct msghdr msg={.msg_iov=&io,.msg_iovlen=1,.msg_control=control,.msg_controllen=sizeof(control)};
    int count=recvmsg(sock,&msg,MSG_CMSG_CLOEXEC),fd=-1;
    if(count>0)for(struct cmsghdr *c=CMSG_FIRSTHDR(&msg);c;c=CMSG_NXTHDR(&msg,c))
        if(c->cmsg_level==SOL_SOCKET && c->cmsg_type==SCM_RIGHTS) memcpy(&fd,CMSG_DATA(c),sizeof(fd));
    close(sock);
    if(fd>=0)return fd;
    errno=ENOENT;
    return -1;
}

static int open_proc(const char *path,int flags,mode_t mode) {
    if(path && !strncmp(path,"/proc/self/task/",16) && strstr(path,"/comm"))
        return syscall(SYS_openat,AT_FDCWD,"/dev/null",flags,mode);
    if(proc_path(path)) {
        char byte;
        return query(path,&byte,1);
    }
    return syscall(SYS_openat,AT_FDCWD,path,flags,mode);
}

int open(const char *path,int flags,...) {
    mode_t mode=0;if(flags&O_CREAT){va_list args;va_start(args,flags);mode=va_arg(args,int);va_end(args);}
    return open_proc(path,flags,mode);
}
int open64(const char *path,int flags,...) {
    mode_t mode=0;if(flags&O_CREAT){va_list args;va_start(args,flags);mode=va_arg(args,int);va_end(args);}
    return open_proc(path,flags,mode);
}
int __open_2(const char *path,int flags){return open_proc(path,flags,0);}
int __open64_2(const char *path,int flags){return open_proc(path,flags,0);}
int openat(int fd,const char *path,int flags,...) {
    mode_t mode=0;if(flags&O_CREAT){va_list args;va_start(args,flags);mode=va_arg(args,int);va_end(args);}
    if(path && path[0]=='/')return open_proc(path,flags,mode);
    return syscall(SYS_openat,fd,path,flags,mode);
}
int openat64(int fd,const char *path,int flags,...) {
    mode_t mode=0;if(flags&O_CREAT){va_list args;va_start(args,flags);mode=va_arg(args,int);va_end(args);}
    if(path && path[0]=='/')return open_proc(path,flags,mode);
    return syscall(SYS_openat,fd,path,flags,mode);
}
FILE *fopen(const char *path,const char *mode) {
    if(path && !strncmp(path,"/proc/self/task/",16) && strstr(path,"/comm"))
        return fdopen(open_proc(path,O_RDWR,0),mode);
    if(proc_path(path)) {
        char byte;int fd=query(path,&byte,1);
        return fd<0?NULL:fdopen(fd,mode);
    }
    static FILE *(*real)(const char*,const char*);
    if(!real)real=dlsym(RTLD_NEXT,"fopen");
    return real(path,mode);
}
FILE *fopen64(const char *path,const char *mode){return fopen(path,mode);}
ssize_t readlink(const char *path,char *buf,size_t size) {
    if(proc_path(path)) {
        char request[512],reply[4096];
        snprintf(request,sizeof(request),"LINK:%s",path);
        int fd=query(request,reply,sizeof(reply));
        if(fd<0)return -1;
        ssize_t count=read(fd,buf,size);close(fd);return count;
    }
    return syscall(SYS_readlinkat,AT_FDCWD,path,buf,size);
}
ssize_t readlinkat(int fd,const char *path,char *buf,size_t size) {
    if(path && path[0]=='/')return readlink(path,buf,size);
    return syscall(SYS_readlinkat,fd,path,buf,size);
}
