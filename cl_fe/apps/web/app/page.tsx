import { AppSidebar } from "@/components/app-sidebar"
import { DashboardPage } from "@/components/dashboard-page"
import {
  SidebarProvider,
  SidebarInset,
  SidebarTrigger,
} from "@workspace/ui/components/sidebar"
import { TooltipProvider } from "@workspace/ui/components/tooltip"
import { Separator } from "@workspace/ui/components/separator"

export default function Page() {
  return (
    <TooltipProvider>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
          <header className="flex h-12 items-center gap-2 border-b px-4">
            <SidebarTrigger />
            <Separator orientation="vertical" className="h-5" />
            <div className="flex items-center gap-2 text-sm">
              <span className="font-medium">Overview</span>
              <span className="text-muted-foreground">·</span>
              <span className="text-muted-foreground text-xs">All Sources</span>
            </div>
          </header>
          <DashboardPage />
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  )
}
