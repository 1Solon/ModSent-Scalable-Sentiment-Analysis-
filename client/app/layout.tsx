import DashboardContent from '../components/page'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <DashboardContent>
          {children}
        </DashboardContent>
      </body>
    </html>
  )
}
