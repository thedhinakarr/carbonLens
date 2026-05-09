"use client"

import {
  IconHome,
  IconSearch,
  IconChartLine,
  IconAdjustments,
  IconNews,
  IconInfoCircle,
  IconFileExport,
  IconSettings,
  IconLeaf,
  IconBell,
  IconHelp,
} from "@tabler/icons-react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from "@workspace/ui/components/sidebar"
import { Avatar, AvatarFallback } from "@workspace/ui/components/avatar"

const mainNav = [
  { title: "Home", icon: IconHome, active: true },
  { title: "Fair Value Lookup", icon: IconSearch },
  { title: "Price Charts", icon: IconChartLine },
  { title: "Quality Explorer", icon: IconAdjustments },
  { title: "News & Sentiment", icon: IconNews, badge: "3" },
]

const secondaryNav = [
  { title: "Reports", icon: IconFileExport },
  { title: "Methodology", icon: IconInfoCircle },
  { title: "Settings", icon: IconSettings },
  { title: "Help", icon: IconHelp },
]

export function AppSidebar() {
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" className="group-data-[collapsible=icon]:p-0!">
              <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <IconLeaf className="size-4" />
              </div>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate text-sm font-semibold">CarbonLens</span>
                <span className="truncate text-xs text-muted-foreground">
                  Pricing Intelligence
                </span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {mainNav.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    isActive={item.active}
                    tooltip={item.title}
                  >
                    <item.icon />
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                  {item.badge && (
                    <SidebarMenuBadge>{item.badge}</SidebarMenuBadge>
                  )}
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarSeparator />

        <SidebarGroup>
          <SidebarGroupLabel>Tools</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {secondaryNav.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton tooltip={item.title}>
                    <item.icon />
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" className="group-data-[collapsible=icon]:p-0!">
              <Avatar className="size-8">
                <AvatarFallback className="bg-primary/20 text-primary text-xs font-semibold">
                  AA
                </AvatarFallback>
              </Avatar>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate text-sm font-medium">Abhinav A.</span>
                <span className="truncate text-xs text-muted-foreground">
                  Admin
                </span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
